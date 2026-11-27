import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import TypeAdapter

from app.api import ws_handler
from app.schemas import contracts as c
from app.services import session_manager as sm
from app.services.nlp_agent import (
    AgentDone,
    AgentEffects,
    AgentTextDelta,
    latest_cv_key,
    spatial_key,
)
from app.services.session_manager import Principal, SessionManager
from app.services.speech_service import MockSpeechProvider, SpeechError, SpeechService
from app.services.vision_service import ImageDecodeError, VisionResult

BARGE_IN_BUDGET_S = 0.1
SERVER_MESSAGE = TypeAdapter(c.ServerMessage)
PCM_FRAME = b"\x01" + b"\x01\x00" * 320  # 0x01 + 20 ms of 16 kHz, 16-bit mono


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.counters: dict[str, int] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self.hashes.setdefault(key, {}).update(mapping)
        return len(mapping)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def delete(self, key: str) -> int:
        existed = key in self.values
        self.values.pop(key, None)
        return int(existed)

    async def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def expire(self, key: str, seconds: int) -> bool:
        self.ttls[key] = seconds
        return True


def make_cart(version: int = 3) -> c.CartSummary:
    item = c.CartItem(
        product_id="prod_001", variant_id="var_001_M", name="Blue Oxford Shirt",
        size="M", quantity=2, unit_price=450000,
    )
    return c.CartSummary(
        cart_id="cart-1",
        version=version,
        items=[item],
        item_count=2,
        total_amount=900000,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.cart = make_cart()
        self.logged: list[dict[str, Any]] = []

    async def authenticate(
        self, session_id: str, token: str | None
    ) -> Principal | None:
        if token == f"good-{session_id}":
            return Principal("user-1", session_id)
        return None

    async def cart_summary(self, user_id: Any) -> c.CartSummary:
        return self.cart

    @asynccontextmanager
    async def gateway(self, user_id: Any) -> AsyncGenerator[Any, None]:
        yield object()

    async def log_interaction(
        self, session_id: str, modality: str, transcript: str | None,
        cv_features: dict[str, Any] | None, response: str, latency_ms: int | None,
    ) -> None:
        self.logged.append(
            {"session_id": session_id, "modality": modality, "transcript": transcript,
             "cv_features": cv_features, "response": response, "latency_ms": latency_ms}
        )


def vision_result(warning: str | None = None) -> VisionResult:
    return VisionResult(
        lab=(24.0, 3.0, -25.0),
        color_name="Navy Blue",
        mean_luminance=20.0 if warning else 120.0,
        quality_warning=warning,
        embedding=[0.1] * 512,
        category_label="shirt",
        category_confidence=0.91,
        is_in_domain=True,
        category_slug="shirts",
    )


class FakeVision:
    """Stands in for VisionService: b"bad" is undecodable, b"dark" is too dark."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[bytes] = []
        self.delay = delay

    async def analyze(self, data: bytes) -> VisionResult:
        self.calls.append(data)
        if self.delay:
            await asyncio.sleep(self.delay)
        if data == b"bad":
            raise ImageDecodeError("no")
        return vision_result("LOW_LIGHT" if data == b"dark" else None)


def deltas(text: str) -> list[Any]:
    """Streamed text the way the agent emits it: a space comes before the next word."""
    words = text.split()
    return [AgentTextDelta(w if i == 0 else " " + w) for i, w in enumerate(words)]


class FakeAgent:
    """Scripted agent. Script items are events, or a float meaning "sleep this long"."""

    def __init__(
        self, script: Callable[[str], list[Any]] | list[Any] | None = None
    ) -> None:
        self._script = script
        self.requests: list[tuple[str, str]] = []
        self.snapshots: list[dict[str, str]] = []
        self.closed = 0
        self.finished_at: float | None = None

    async def respond(
        self,
        *,
        gateway: Any,
        redis: Any,
        session_id: str,
        user_text: str,
        language: str,
    ) -> AsyncGenerator[Any, None]:
        self.requests.append((user_text, language))
        self.snapshots.append(dict(redis.values))
        if callable(self._script):
            script = self._script(user_text)
        elif self._script is not None:
            script = self._script
        else:
            reply = "Chào bạn. Mình giúp gì được ạ?"
            script = [*deltas(reply), AgentDone(reply, [], None)]
        try:
            for item in script:
                if isinstance(item, float):
                    await asyncio.sleep(item)
                else:
                    yield item
            self.finished_at = time.perf_counter()
        finally:
            self.closed += 1


class FakeWebSocket:
    """The parts of a Starlette WebSocket that `serve` uses, driven by the test."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.sent_at: list[float] = []
        # Sampled whenever a message is sent.
        self.probe: Callable[[], Any] | None = None
        self.probe_values: list[Any] = []
        self.accepted = False
        self.close_code: int | None = None
        self._cursor = 0

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, Any]:
        return await self.incoming.get()

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))
        self.sent_at.append(time.perf_counter())
        self.probe_values.append(self.probe() if self.probe else None)

    async def close(self, code: int = 1000) -> None:
        self.close_code = code
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": code})

    # client side
    def client_send(self, message: dict[str, Any]) -> None:
        text = json.dumps(message)
        self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    def client_send_raw(self, text: str) -> None:
        self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    def client_send_bytes(self, data: bytes) -> None:
        self.incoming.put_nowait({"type": "websocket.receive", "bytes": data})

    def client_disconnect(self) -> None:
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m["type"] == kind]

    async def next(
        self, kind: str | None = None, timeout: float = 2.0
    ) -> dict[str, Any]:
        """The next unread message (of `kind`, skipping others)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            while self._cursor < len(self.sent):
                message = self.sent[self._cursor]
                self._cursor += 1
                if kind is None or message["type"] == kind:
                    return message
            await asyncio.sleep(0.005)
        wanted = kind or "a message"
        raise AssertionError(f"timed out waiting for {wanted}; got {self.sent}")


async def wait_until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met in time")


@dataclass
class Env:
    manager: SessionManager
    redis: FakeRedis
    backend: FakeBackend
    vision: FakeVision
    agent: FakeAgent
    provider: MockSpeechProvider
    sockets: list[tuple[FakeWebSocket, asyncio.Task[Any]]] = field(default_factory=list)


def make_env(
    agent: FakeAgent | None = None,
    provider: MockSpeechProvider | None = None,
    vision: FakeVision | None = None,
    **manager_kwargs: Any,
) -> Env:
    redis, backend = FakeRedis(), FakeBackend()
    vision = vision or FakeVision()
    agent = agent or FakeAgent()
    provider = provider or MockSpeechProvider()
    manager = SessionManager(
        redis=redis, backend=backend, vision=vision, speech=SpeechService(provider),
        agent_provider=lambda: agent, **manager_kwargs,
    )
    return Env(manager, redis, backend, vision, agent, provider)


async def connect(
    env: Env, session_id: str = "s1", token: str | None = "default",
    headers: dict[str, str] | None = None,
) -> tuple[FakeWebSocket, asyncio.Task[Any]]:
    websocket = FakeWebSocket(headers)
    actual = f"good-{session_id}" if token == "default" else token
    serving = ws_handler.serve(websocket, session_id, actual, env.manager)
    task = asyncio.ensure_future(serving)
    env.sockets.append((websocket, task))
    return websocket, task


async def hang_up(websocket: FakeWebSocket, task: asyncio.Task[Any]) -> None:
    websocket.client_disconnect()
    await asyncio.wait_for(task, 3)
    for message in websocket.sent:  # every server message must match the schema
        SERVER_MESSAGE.validate_python(message)


def msg(kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": kind, "timestamp": 1795015500123, "payload": payload or {}}


def text_message(
    text: str, want_audio: bool = False, language: str = "vi"
) -> dict[str, Any]:
    payload = {"text": text, "language": language, "want_audio": want_audio}
    return msg("TEXT_MESSAGE", payload)


def cv_message(image: bytes, gaze: str | None = None) -> dict[str, Any]:
    return msg(
        "CV_FRAME_ROI",
        {"image_base64": base64.b64encode(image).decode(), "hand_confidence": 0.9,
         "gaze_target_product_id": gaze},
    )


def telemetry(
    aisle: str | None = "A1", gaze: str | None = "prod_001"
) -> dict[str, Any]:
    return msg(
        "PLAYER_TELEMETRY",
        {"position": {"x": 4.2, "y": 0.0, "z": 1.8}, "yaw": 90.0, "pitch": -5.0,
         "current_aisle": aisle, "gaze_target_product_id": gaze},
    )


def chunks_of(
    websocket: FakeWebSocket, response_id: str | None = None
) -> list[dict[str, Any]]:
    found = [m["payload"] for m in websocket.of_type("AGENT_RESPONSE_STREAM")]
    return [p for p in found if response_id is None or p["response_id"] == response_id]


async def wait_final(websocket: FakeWebSocket, timeout: float = 3.0) -> str:
    await wait_until(
        lambda: any(p["is_final"] for p in chunks_of(websocket)), timeout
    )
    return next(p["response_id"] for p in chunks_of(websocket) if p["is_final"])


# ---------------------------------------------------------------------------
# Connection, authentication, sync
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", [None, "garbage", "good-another-session"])
async def test_bad_tokens_are_closed_with_4401(token: str | None) -> None:
    env = make_env()
    websocket, task = await connect(env, token=token)
    await asyncio.wait_for(task, 2)
    assert websocket.accepted is True
    assert websocket.close_code == 4401
    assert websocket.sent == []  # nothing is revealed to an unauthenticated client


async def test_sync_is_the_first_message_with_cart_position_and_waypoint() -> None:
    env = make_env()
    position = {"pos_x": "4.2", "pos_y": "0", "pos_z": "1.8", "current_aisle": "A1"}
    await env.redis.hset(spatial_key("s1"), position)
    await env.redis.set(sm.last_waypoint_key("s1"), "WP_AISLE_01_SHIRTS")
    websocket, task = await connect(env)

    first = await websocket.next()
    assert first["type"] == "SESSION_SYNC"
    payload = first["payload"]
    assert payload["cart"]["version"] == 3 and payload["cart"]["item_count"] == 2
    assert payload["spatial"] == {
        "position": {"x": 4.2, "y": 0.0, "z": 1.8},
        "current_aisle": "A1",
    }
    assert payload["last_waypoint_id"] == "WP_AISLE_01_SHIRTS"
    await hang_up(websocket, task)


async def test_reconnect_restores_cart_position_and_waypoint() -> None:
    env = make_env()
    first, first_task = await connect(env)
    await first.next("SESSION_SYNC")
    first.client_send(telemetry("A2", None))
    await wait_until(lambda: spatial_key("s1") in env.redis.hashes)
    env.backend.cart = make_cart(version=7)
    await env.redis.set(sm.last_waypoint_key("s1"), "WP_AISLE_02_JACKETS")
    await hang_up(first, first_task)

    second, second_task = await connect(env)
    sync = (await second.next("SESSION_SYNC"))["payload"]
    assert sync["cart"]["version"] == 7
    assert sync["spatial"]["current_aisle"] == "A2"
    assert sync["spatial"]["position"]["x"] == 4.2
    assert sync["last_waypoint_id"] == "WP_AISLE_02_JACKETS"
    await hang_up(second, second_task)


async def test_resync_request_sends_the_state_again() -> None:
    env = make_env()
    websocket, task = await connect(env)
    await websocket.next("SESSION_SYNC")
    websocket.client_send(msg("RESYNC_REQUEST"))
    assert (await websocket.next("SESSION_SYNC"))["type"] == "SESSION_SYNC"
    await hang_up(websocket, task)


async def test_heartbeat_is_echoed() -> None:
    env = make_env()
    websocket, task = await connect(env)
    await websocket.next("SESSION_SYNC")
    websocket.client_send(msg("HEARTBEAT"))
    assert (await websocket.next("HEARTBEAT"))["payload"] == {}
    await hang_up(websocket, task)


async def test_a_silent_connection_is_closed_with_4408() -> None:
    env = make_env(idle_timeout_s=0.15)
    websocket, task = await connect(env)
    await asyncio.wait_for(task, 2)
    assert websocket.close_code == 4408


async def test_heartbeats_keep_the_connection_alive() -> None:
    env = make_env(idle_timeout_s=0.3)
    websocket, task = await connect(env)
    for _ in range(4):
        await asyncio.sleep(0.15)
        websocket.client_send(msg("HEARTBEAT"))
    assert not task.done()
    await hang_up(websocket, task)


async def test_disallowed_origin_is_closed_with_4403() -> None:
    env = make_env(allowed_origins=("https://shop.example",))
    bad, bad_task = await connect(env, headers={"origin": "https://evil.example"})
    await asyncio.wait_for(bad_task, 2)
    assert bad.close_code == 4403

    good, good_task = await connect(env, headers={"origin": "https://shop.example"})
    assert (await good.next())["type"] == "SESSION_SYNC"
    await hang_up(good, good_task)


async def test_connections_per_user_are_limited() -> None:
    env = make_env(max_connections_per_user=2)
    first, first_task = await connect(env)
    second, second_task = await connect(env)
    await first.next("SESSION_SYNC")
    await second.next("SESSION_SYNC")

    third, third_task = await connect(env)
    await asyncio.wait_for(third_task, 2)
    assert third.close_code == 4429

    await hang_up(first, first_task)
    fourth, fourth_task = await connect(env)
    assert (await fourth.next())["type"] == "SESSION_SYNC"
    await hang_up(second, second_task)
    await hang_up(fourth, fourth_task)


# ---------------------------------------------------------------------------
# Text turns
# ---------------------------------------------------------------------------


async def test_text_message_gets_a_streamed_text_reply() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(text_message("Xin chào"))
    response_id = await wait_final(websocket)

    chunks = chunks_of(websocket, response_id)
    assert response_id == "resp_0001"
    assert [p["chunk_seq"] for p in chunks] == list(range(len(chunks)))
    assert [p["is_final"] for p in chunks] == [False] * (len(chunks) - 1) + [True]
    spoken = " ".join(p["text"] for p in chunks if p["text"])
    assert spoken == "Chào bạn. Mình giúp gì được ạ?"
    for payload in chunks:  # a text-only reply has no audio and no visemes
        assert payload["audio_base64"] is None and payload["audio_format"] is None
        assert payload["audio_duration_ms"] == 0 and payload["visemes"] == []
    assert env.agent.requests == [("Xin chào", "vi")]
    await hang_up(websocket, task)


async def test_text_message_with_audio_carries_valid_visemes() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(text_message("Xin chào", want_audio=True))
    await wait_final(websocket)

    for payload in chunks_of(websocket):
        if not payload["text"]:
            continue
        assert base64.b64decode(payload["audio_base64"]).startswith(b"MOCK-MP3:")
        assert payload["audio_format"] == "mp3" and payload["audio_duration_ms"] > 0
        times = [v["time_ms"] for v in payload["visemes"]]
        assert times and times[0] == 0
        assert times == sorted(set(times)) and times[-1] <= payload["audio_duration_ms"]
    await hang_up(websocket, task)


async def test_each_sentence_is_sent_as_soon_as_it_is_complete() -> None:
    script = [
        AgentTextDelta("Câu một."),
        AgentTextDelta(" Câu"),
        0.4,
        AgentTextDelta(" hai."),
        AgentDone("Câu một. Câu hai.", [], None),
    ]
    agent = FakeAgent(script)
    env = make_env(agent)
    websocket, task = await connect(env)
    websocket.client_send(text_message("hi"))
    await wait_final(websocket)

    first_index = next(
        i for i, m in enumerate(websocket.sent) if m["type"] == "AGENT_RESPONSE_STREAM"
    )
    first_chunk_at = websocket.sent_at[first_index]
    assert agent.finished_at is not None
    # The first sentence went out while the agent was still working.
    assert first_chunk_at < agent.finished_at - 0.2
    assert [p["text"] for p in chunks_of(websocket)] == ["Câu một.", "Câu hai."]
    await hang_up(websocket, task)


async def test_final_chunk_carries_products_and_animation() -> None:
    reply = "Mình dẫn bạn tới áo sơ mi nhé."
    done = AgentDone(reply, ["prod_001", "prod_004"], "NOD")
    agent = FakeAgent([*deltas(reply), done])
    env = make_env(agent)
    websocket, task = await connect(env)
    websocket.client_send(text_message("dẫn tôi đi"))
    await wait_final(websocket)

    (chunk,) = chunks_of(websocket)
    assert chunk["is_final"] is True and chunk["chunk_seq"] == 0
    assert chunk["recommended_products"] == ["prod_001", "prod_004"]
    assert chunk["avatar_animation"] == "NOD"
    await hang_up(websocket, task)


async def test_text_that_already_ended_gets_an_empty_final_chunk() -> None:
    done = AgentDone("Một. Hai.", ["prod_002"], None)
    agent = FakeAgent([AgentTextDelta("Một. Hai. "), done])
    env = make_env(agent)
    websocket, task = await connect(env)
    websocket.client_send(text_message("hi"))
    await wait_final(websocket)

    chunks = chunks_of(websocket)
    assert [p["text"] for p in chunks] == ["Một.", "Hai.", ""]
    assert chunks[-1]["is_final"] is True
    assert chunks[-1]["recommended_products"] == ["prod_002"]
    await hang_up(websocket, task)


def navigation_effects() -> AgentEffects:
    commands = [
        c.PlayAvatarAnimationCommand(
            action="PLAY_AVATAR_ANIMATION",
            payload=c.PlayAvatarAnimationPayload(animation="GESTURE_GUIDE"),
        ),
        c.NavigateToWaypointCommand(
            action="NAVIGATE_TO_WAYPOINT",
            payload=c.NavigateToWaypointPayload(waypoint_id="WP_AISLE_01_SHIRTS"),
        ),
        c.HighlightProductsCommand(
            action="HIGHLIGHT_PRODUCTS",
            payload=c.HighlightProductsPayload(product_ids=["prod_001"]),
        ),
    ]
    return AgentEffects(
        commands=commands,
        ui_action=c.OpenCheckoutAction(action="OPEN_CHECKOUT"),
        cart=make_cart(version=4),
    )


async def test_tool_effects_become_unity_ui_and_cart_messages() -> None:
    reply = "Mình dẫn bạn tới áo sơ mi nhé."
    done = AgentDone(reply, ["prod_001"], "GESTURE_GUIDE")
    agent = FakeAgent([navigation_effects(), *deltas(reply), done])
    env = make_env(agent)
    websocket, task = await connect(env)
    websocket.client_send(text_message("dẫn tôi tới áo sơ mi"))
    response_id = await wait_final(websocket)

    kinds = [m["type"] for m in websocket.sent[1:]]  # after SESSION_SYNC
    assert kinds[:3] == ["UNITY_COMMAND", "UI_ACTION", "CART_UPDATED"]
    unity = websocket.of_type("UNITY_COMMAND")[0]["payload"]
    assert unity["response_id"] == response_id
    assert [cmd["action"] for cmd in unity["commands"]] == [
        "PLAY_AVATAR_ANIMATION", "NAVIGATE_TO_WAYPOINT", "HIGHLIGHT_PRODUCTS",
    ]
    assert websocket.of_type("UI_ACTION")[0]["payload"] == {"action": "OPEN_CHECKOUT"}
    assert websocket.of_type("CART_UPDATED")[0]["payload"]["version"] == 4
    assert env.redis.values[sm.last_waypoint_key("s1")] == "WP_AISLE_01_SHIRTS"
    assert chunks_of(websocket)[0]["avatar_animation"] == "GESTURE_GUIDE"
    await hang_up(websocket, task)


async def test_interaction_is_logged_without_images_or_embeddings() -> None:
    env = make_env()
    await env.redis.set(
        latest_cv_key("s1"),
        json.dumps(
            {
                "lab": [1, 2, 3],
                "color_name": "Navy Blue",
                "category_label": "shirt",
                "category_confidence": 0.9,
                "embedding": [0.1] * 512,
            }
        ),
    )
    websocket, task = await connect(env)
    websocket.client_send(text_message("Xin chào"))
    await wait_final(websocket)
    await wait_until(lambda: bool(env.backend.logged))

    (entry,) = env.backend.logged
    assert entry["modality"] == "TEXT" and entry["transcript"] == "Xin chào"
    assert entry["response"] == "Chào bạn. Mình giúp gì được ạ?"
    assert entry["cv_features"] == {
        "color_name": "Navy Blue",
        "category_label": "shirt",
        "category_confidence": 0.9,
    }
    assert entry["latency_ms"] is not None and entry["latency_ms"] >= 0
    await hang_up(websocket, task)


async def test_tts_failure_falls_back_to_a_text_chunk() -> None:
    provider = MockSpeechProvider(tts_error=SpeechError("RATE_LIMITED", "slow down"))
    env = make_env(provider=provider)
    websocket, task = await connect(env)
    websocket.client_send(text_message("hi", want_audio=True))
    await wait_final(websocket)

    error = websocket.of_type("ERROR")[0]["payload"]
    assert error["code"] == "RATE_LIMITED" and error["recoverable"] is True
    chunk = chunks_of(websocket)[-1]
    assert chunk["is_final"] is True and chunk["audio_base64"] is None and chunk["text"]
    await hang_up(websocket, task)


async def test_response_ids_continue_after_a_reconnect() -> None:
    env = make_env()
    first, first_task = await connect(env)
    first.client_send(text_message("một"))
    assert await wait_final(first) == "resp_0001"
    await hang_up(first, first_task)

    second, second_task = await connect(env)
    second.client_send(text_message("hai"))
    assert await wait_final(second) == "resp_0002"
    await hang_up(second, second_task)


# ---------------------------------------------------------------------------
# Barge-in
# ---------------------------------------------------------------------------


def slow_script() -> list[Any]:
    return [AgentTextDelta("Câu một."), AgentTextDelta(" Câu hai."), 0.5,
            AgentTextDelta(" Câu ba."), 0.5, AgentTextDelta(" Câu bốn."),
            AgentDone("Câu một. Câu hai. Câu ba. Câu bốn.", [], None)]


async def test_barge_in_cancels_within_100_ms_and_no_chunk_follows() -> None:
    agent = FakeAgent(slow_script())
    env = make_env(agent)
    websocket, task = await connect(env)
    websocket.client_send(text_message("kể chuyện"))
    first = await websocket.next("AGENT_RESPONSE_STREAM")
    response_id = first["payload"]["response_id"]
    assert env.redis.values[sm.active_response_key("s1")] == response_id

    started = time.perf_counter()
    websocket.client_send(msg("BARGE_IN_INTERRUPT", {"response_id": response_id}))
    cancelled = await websocket.next("RESPONSE_CANCELLED")
    elapsed = time.perf_counter() - started

    assert cancelled["payload"]["response_id"] == response_id
    assert elapsed < BARGE_IN_BUDGET_S
    marker = websocket.sent.index(cancelled)
    await asyncio.sleep(0.8)  # long enough for the old script to have produced more
    after = websocket.sent[marker + 1 :]
    later = [m for m in after if m["type"] == "AGENT_RESPONSE_STREAM"]
    assert [m for m in later if m["payload"]["response_id"] == response_id] == []
    assert agent.closed == 1  # the agent's stream was closed
    assert sm.active_response_key("s1") not in env.redis.values
    assert env.backend.logged == []  # a cancelled response is not logged as an answer
    await hang_up(websocket, task)


async def test_cancelled_is_only_sent_after_the_response_task_has_stopped() -> None:
    agent = FakeAgent(slow_script())
    env = make_env(agent)
    websocket, task = await connect(env)
    # How many agent streams had already ended when each message was sent.
    websocket.probe = lambda: agent.closed
    websocket.client_send(text_message("kể chuyện"))
    first = await websocket.next("AGENT_RESPONSE_STREAM")

    websocket.client_send(
        msg("BARGE_IN_INTERRUPT", {"response_id": first["payload"]["response_id"]})
    )
    cancelled = await websocket.next("RESPONSE_CANCELLED")
    assert websocket.probe_values[websocket.sent.index(cancelled)] == 1
    await hang_up(websocket, task)


class StubbornAgent(FakeAgent):
    """An agent that ignores cancellation and keeps producing sentences."""

    async def respond(
        self,
        *,
        gateway: Any,
        redis: Any,
        session_id: str,
        user_text: str,
        language: str,
    ) -> AsyncGenerator[Any, None]:
        yield AgentTextDelta("Câu đầu. ")
        for number in range(6):
            try:
                await asyncio.sleep(0.15)
            except asyncio.CancelledError:
                pass  # refuses to stop
            yield AgentTextDelta(f"Tiếp {number}. ")
        yield AgentDone("x", [], None)


async def test_cancelled_response_never_sends_even_if_it_keeps_running() -> None:
    env = make_env(StubbornAgent())
    websocket, task = await connect(env)
    websocket.client_send(text_message("kể chuyện"))
    first = await websocket.next("AGENT_RESPONSE_STREAM")
    response_id = first["payload"]["response_id"]
    sent_before = len(websocket.sent)

    websocket.client_send(msg("BARGE_IN_INTERRUPT", {"response_id": response_id}))
    await websocket.next("RESPONSE_CANCELLED")
    await asyncio.sleep(0.5)  # the agent is still producing sentences in the background
    leaked = [
        m
        for m in websocket.sent[sent_before:]
        if m["type"] == "AGENT_RESPONSE_STREAM"
        and m["payload"]["response_id"] == response_id
    ]
    assert leaked == []
    env.sockets.clear()
    await hang_up(websocket, task)


async def test_barge_in_for_an_unknown_response_is_just_confirmed() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(msg("BARGE_IN_INTERRUPT", {"response_id": "resp_9999"}))
    cancelled = await websocket.next("RESPONSE_CANCELLED")
    assert cancelled["payload"]["response_id"] == "resp_9999"
    await hang_up(websocket, task)


async def test_a_new_message_cancels_the_active_response() -> None:
    answer = AgentDone("Trả lời mới.", [], None)
    scripts = {"một": slow_script(), "hai": [*deltas("Trả lời mới."), answer]}
    agent = FakeAgent(lambda text: scripts[text])
    env = make_env(agent)
    websocket, task = await connect(env)
    websocket.client_send(text_message("một"))
    first = (await websocket.next("AGENT_RESPONSE_STREAM"))["payload"]["response_id"]

    websocket.client_send(text_message("hai"))
    cancelled = await websocket.next("RESPONSE_CANCELLED")
    assert cancelled["payload"]["response_id"] == first
    second = await wait_final(websocket)
    assert second != first

    marker = websocket.sent.index(cancelled)
    await asyncio.sleep(0.7)
    after = [m["payload"]["response_id"] for m in websocket.sent[marker + 1 :]
             if m["type"] == "AGENT_RESPONSE_STREAM"]
    assert first not in after and second in after
    await hang_up(websocket, task)


async def test_disconnecting_cancels_running_work_and_frees_the_slot() -> None:
    agent = FakeAgent(slow_script())
    env = make_env(agent, max_connections_per_user=1)
    websocket, task = await connect(env)
    websocket.client_send(text_message("kể chuyện"))
    await websocket.next("AGENT_RESPONSE_STREAM")

    await hang_up(websocket, task)
    assert agent.closed == 1
    assert sm.active_response_key("s1") not in env.redis.values
    again, again_task = await connect(env)  # the single slot is free again
    assert (await again.next())["type"] == "SESSION_SYNC"
    await hang_up(again, again_task)


# ---------------------------------------------------------------------------
# Voice turns
# ---------------------------------------------------------------------------


async def test_voice_turn_answers_with_audio_and_logs_latency() -> None:
    provider = MockSpeechProvider(transcript="Tìm áo giống màu này", confidence=0.93)
    env = make_env(provider=provider)
    websocket, task = await connect(env)
    for _ in range(5):
        websocket.client_send_bytes(PCM_FRAME)
    websocket.client_send(msg("AUDIO_END", {"audio_mode": "ptt"}))
    await wait_final(websocket)

    transcripts = [m["payload"] for m in websocket.of_type("STT_TRANSCRIPT")]
    assert transcripts[0]["is_final"] is False
    assert transcripts[-1] == {"text": "Tìm áo giống màu này", "confidence": 0.93,
                               "is_final": True, "language": "vi"}
    assert env.agent.requests == [("Tìm áo giống màu này", "vi")]
    assert len(provider.sessions[0].frames) == 5
    assert all(p["audio_base64"] for p in chunks_of(websocket) if p["text"])

    await wait_until(lambda: bool(env.backend.logged))
    latency = env.backend.logged[0]["latency_ms"]
    assert env.backend.logged[0]["modality"] == "VOICE"
    assert 0 <= latency < sm.LATENCY_BUDGET_MS
    assert list(env.manager.latencies_ms) == [latency]
    await hang_up(websocket, task)


@pytest.mark.parametrize(
    "frame",
    [b"", b"\x01", b"\x02" + b"\x00" * 640, b"\x01" + b"\x00" * 641,
     b"\x01" + b"\x00" * (sm.MAX_AUDIO_FRAME_BYTES + 2)],
    ids=["empty", "prefix only", "wrong prefix", "odd length", "too large"],
)
async def test_invalid_audio_frames_are_ignored(frame: bytes) -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send_bytes(frame)
    websocket.client_send(msg("HEARTBEAT"))
    await websocket.next("HEARTBEAT")
    assert env.provider.sessions == []  # no speech recognition was started
    await hang_up(websocket, task)


async def test_audio_end_without_audio_does_nothing() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(msg("AUDIO_END", {"audio_mode": "vad"}))
    websocket.client_send(msg("HEARTBEAT"))
    await websocket.next("HEARTBEAT")
    assert env.agent.requests == [] and websocket.of_type("ERROR") == []
    await hang_up(websocket, task)


async def test_unrecognised_speech_reports_stt_failed() -> None:
    env = make_env(provider=MockSpeechProvider(transcript=""))
    websocket, task = await connect(env)
    websocket.client_send_bytes(PCM_FRAME)
    websocket.client_send(msg("AUDIO_END", {"audio_mode": "ptt"}))
    error = (await websocket.next("ERROR"))["payload"]
    assert error["code"] == "STT_FAILED" and error["recoverable"] is True
    assert env.agent.requests == []
    await hang_up(websocket, task)


async def test_speech_provider_errors_are_reported() -> None:
    failure = SpeechError("STT_FAILED", "recogniser down")
    provider = MockSpeechProvider(stt_error=failure)
    env = make_env(provider=provider)
    websocket, task = await connect(env)
    websocket.client_send_bytes(PCM_FRAME)
    websocket.client_send(msg("AUDIO_END", {"audio_mode": "ptt"}))
    error = (await websocket.next("ERROR"))["payload"]
    assert error == {
        "code": "STT_FAILED",
        "message": "recogniser down",
        "recoverable": True,
    }
    await hang_up(websocket, task)


async def test_a_second_utterance_after_audio_end_is_a_separate_turn() -> None:
    env = make_env(provider=MockSpeechProvider(transcript="câu hỏi"))
    websocket, task = await connect(env)
    for turn in range(2):
        websocket.client_send_bytes(PCM_FRAME)
        websocket.client_send(msg("AUDIO_END", {"audio_mode": "ptt"}))
        await wait_until(lambda t=turn: len(env.agent.requests) == t + 1)
        await wait_until(
            lambda t=turn: sum(p["is_final"] for p in chunks_of(websocket)) == t + 1
        )
    assert len(env.provider.sessions) == 2
    await hang_up(websocket, task)


# ---------------------------------------------------------------------------
# Camera and telemetry
# ---------------------------------------------------------------------------


async def test_cv_frame_is_analysed_and_only_features_are_stored() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(cv_message(b"good-image", gaze="prod_002"))
    await wait_until(lambda: latest_cv_key("s1") in env.redis.values)

    stored = json.loads(env.redis.values[latest_cv_key("s1")])
    assert set(stored) == {"lab", "color_name", "category_label", "category_confidence",
                           "quality_warning", "embedding", "captured_at"}
    assert stored["color_name"] == "Navy Blue" and len(stored["embedding"]) == 512
    assert env.redis.ttls[latest_cv_key("s1")] == sm.LATEST_CV_TTL_S == 30
    assert env.vision.calls == [b"good-image"]
    assert base64.b64encode(b"good-image").decode() not in json.dumps(env.redis.values)
    assert env.redis.hashes[spatial_key("s1")]["gaze_product_id"] == "prod_002"
    await hang_up(websocket, task)


async def test_oversized_cv_frame_is_dropped() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(cv_message(b"x" * (sm.MAX_CV_BYTES + 1000)))
    websocket.client_send(msg("HEARTBEAT"))
    await websocket.next("HEARTBEAT")
    await asyncio.sleep(0.05)
    assert env.vision.calls == [] and websocket.of_type("ERROR") == []
    await hang_up(websocket, task)


async def test_a_frame_exactly_at_the_limit_is_accepted() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(cv_message(b"x" * sm.MAX_CV_BYTES))
    await wait_until(lambda: len(env.vision.calls) == 1)
    await hang_up(websocket, task)


async def test_cv_frames_are_limited_to_two_per_second() -> None:
    env = make_env()
    websocket, task = await connect(env)
    for index in range(8):
        websocket.client_send(cv_message(f"frame-{index}".encode()))
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    assert len(env.vision.calls) == 2
    await hang_up(websocket, task)


async def test_cv_frame_arriving_during_analysis_is_skipped() -> None:
    env = make_env(vision=FakeVision(delay=0.2))
    websocket, task = await connect(env)
    websocket.client_send(cv_message(b"first"))
    await asyncio.sleep(0.02)
    websocket.client_send(cv_message(b"second"))
    await asyncio.sleep(0.4)
    assert env.vision.calls == [b"first"]
    await hang_up(websocket, task)


@pytest.mark.parametrize("image", [b"bad", None], ids=["undecodable", "not base64"])
async def test_unreadable_cv_frames_report_an_error(image: bytes | None) -> None:
    env = make_env()
    websocket, task = await connect(env)
    message = cv_message(image or b"")
    if image is None:
        message["payload"]["image_base64"] = "***not base64***"
    websocket.client_send(message)
    error = (await websocket.next("ERROR"))["payload"]
    assert error["code"] == "CV_UNSUPPORTED_IMAGE" and error["recoverable"] is True
    await hang_up(websocket, task)


async def test_low_light_warning_is_sent_once_not_for_every_frame() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(cv_message(b"dark"))
    warning = (await websocket.next("CV_QUALITY_WARNING"))["payload"]
    assert warning["warning"] == "LOW_LIGHT" and 0 <= warning["mean_luminance"] <= 255

    websocket.client_send(cv_message(b"dark"))
    await wait_until(lambda: len(env.vision.calls) == 2)
    await asyncio.sleep(0.05)
    assert len(websocket.of_type("CV_QUALITY_WARNING")) == 1
    await hang_up(websocket, task)


async def test_camera_and_audio_are_handled_at_the_same_time() -> None:
    env = make_env(provider=MockSpeechProvider(transcript="tìm cái giống cái này"))
    websocket, task = await connect(env)
    websocket.client_send_bytes(PCM_FRAME)  # the shopper starts talking...
    websocket.client_send_bytes(PCM_FRAME)
    await wait_until(lambda: len(env.provider.sessions) == 1)
    websocket.client_send(cv_message(b"good-image"))  # ...while a camera frame arrives
    await wait_until(lambda: latest_cv_key("s1") in env.redis.values)
    assert env.provider.sessions[0].closed is False  # speech recognition is still open

    websocket.client_send(msg("AUDIO_END", {"audio_mode": "vad"}))
    await wait_final(websocket)
    # The agent saw the camera result.
    assert latest_cv_key("s1") in env.agent.snapshots[0]
    await hang_up(websocket, task)


async def test_telemetry_is_written_to_redis() -> None:
    env = make_env()
    websocket, task = await connect(env)
    websocket.client_send(telemetry("A1", "prod_001"))
    await wait_until(lambda: spatial_key("s1") in env.redis.hashes)
    await asyncio.sleep(0.02)
    spatial = env.redis.hashes[spatial_key("s1")]
    assert (spatial["pos_x"], spatial["pos_z"]) == ("4.2", "1.8")
    assert (spatial["yaw"], spatial["pitch"]) == ("90.0", "-5.0")
    assert spatial["current_aisle"] == "A1" and spatial["gaze_product_id"] == "prod_001"
    assert int(spatial["updated_at"]) > 0
    assert env.redis.ttls[spatial_key("s1")] == 3600

    websocket.client_send(telemetry(None, None))
    spatial_hash = env.redis.hashes[spatial_key("s1")]
    await wait_until(lambda: spatial_hash["gaze_product_id"] == "")
    assert env.redis.hashes[spatial_key("s1")]["current_aisle"] == ""
    await hang_up(websocket, task)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


async def test_bad_messages_get_an_error_and_the_session_survives() -> None:
    env = make_env()
    websocket, task = await connect(env)
    await websocket.next("SESSION_SYNC")
    websocket.client_send_raw("this is not json")
    websocket.client_send(msg("DO_SOMETHING_ELSE"))
    websocket.client_send({"type": "TEXT_MESSAGE", "timestamp": 1, "payload": {}})
    bad_time = {"type": "HEARTBEAT", "timestamp": "yesterday", "payload": {}}
    websocket.client_send(bad_time)
    for _ in range(4):
        error = (await websocket.next("ERROR"))["payload"]
        assert error["recoverable"] is True

    websocket.client_send(msg("HEARTBEAT"))
    assert (await websocket.next("HEARTBEAT"))["type"] == "HEARTBEAT"
    await hang_up(websocket, task)


def test_take_sentences_splits_only_finished_sentences() -> None:
    assert sm.take_sentences("Xin chào. Bạn cần gì") == (["Xin chào."], "Bạn cần gì")
    assert sm.take_sentences("Giá 450.000đ rất rẻ") == ([], "Giá 450.000đ rất rẻ")
    assert sm.take_sentences("A. B? C! ") == (["A.", "B?", "C!"], "")
    assert sm.take_sentences("Dòng một\nDòng hai") == (["Dòng một"], "Dòng hai")
    assert sm.take_sentences("") == ([], "")


# ---------------------------------------------------------------------------
# The real FastAPI route (Starlette TestClient)
# ---------------------------------------------------------------------------


def make_app(env: Env) -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(ws_handler.router)
    app.state.session_manager = env.manager
    return app


def test_real_route_rejects_a_bad_token_testclient() -> None:
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    with TestClient(make_app(make_env())) as client:
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect("/ws/v1/session/s1?token=nope") as websocket:
                websocket.receive_json()
        assert caught.value.code == 4401


def test_real_route_syncs_and_echoes_heartbeats_testclient() -> None:
    from fastapi.testclient import TestClient

    with TestClient(make_app(make_env())) as client:
        with client.websocket_connect("/ws/v1/session/s1?token=good-s1") as websocket:
            sync = websocket.receive_json()
            SERVER_MESSAGE.validate_python(sync)
            assert sync["type"] == "SESSION_SYNC"
            websocket.send_json(msg("HEARTBEAT"))
            assert websocket.receive_json()["type"] == "HEARTBEAT"


def test_real_route_handles_binary_audio_and_text_testclient() -> None:
    from fastapi.testclient import TestClient

    env = make_env(provider=MockSpeechProvider(transcript="xin chào"))
    with TestClient(make_app(env)) as client:
        with client.websocket_connect("/ws/v1/session/s1?token=good-s1") as websocket:
            websocket.receive_json()
            websocket.send_bytes(PCM_FRAME)
            websocket.send_json(msg("AUDIO_END", {"audio_mode": "ptt"}))
            seen = []
            while True:
                message = websocket.receive_json()
                seen.append(message["type"])
                is_chunk = message["type"] == "AGENT_RESPONSE_STREAM"
                if is_chunk and message["payload"]["is_final"]:
                    break
        assert "STT_TRANSCRIPT" in seen


# ---------------------------------------------------------------------------
# The SQL backend (needs the test database)
# ---------------------------------------------------------------------------


@pytest.fixture
def sql_backend(db_engine: Any) -> Any:
    from app.core.database import create_session_factory

    return sm.SqlBackend(create_session_factory(db_engine))


async def test_sql_backend_authenticates_only_the_right_token_and_session(
    sql_backend: Any, db_session: Any
) -> None:
    from app.core.config import get_settings
    from app.services import auth_service

    settings = get_settings()
    login = await auth_service.start_guest_session(db_session, False, settings)

    principal = await sql_backend.authenticate(login.session_id, login.access_token)
    assert principal is not None and str(principal.user_id) == login.user_id

    assert await sql_backend.authenticate("other-session", login.access_token) is None
    assert await sql_backend.authenticate(login.session_id, "garbage") is None
    assert await sql_backend.authenticate(login.session_id, None) is None
    expired = auth_service.create_access_token(
        principal.user_id, login.session_id, settings, ttl_seconds=-10
    )
    assert await sql_backend.authenticate(login.session_id, expired) is None
    stranger = auth_service.create_access_token(
        __import__("uuid").uuid4(), login.session_id, settings
    )
    assert await sql_backend.authenticate(login.session_id, stranger) is None


async def test_sql_backend_cart_and_interaction_log(
    sql_backend: Any, db_session: Any
) -> None:
    from sqlalchemy import select

    from app.core.config import get_settings
    from app.models.domain import InteractionLog
    from app.services import auth_service

    login = await auth_service.start_guest_session(db_session, False, get_settings())
    principal = await sql_backend.authenticate(login.session_id, login.access_token)

    cart = await sql_backend.cart_summary(principal.user_id)
    assert (cart.item_count, cart.total_amount, cart.version) == (0, 0, 0)

    await sql_backend.log_interaction(
        login.session_id, "VOICE", "xin chào", {"color_name": "Red"}, "Chào bạn", 420
    )
    rows = (await db_session.execute(select(InteractionLog))).scalars().all()
    assert [(r.modality, r.user_transcript, r.latency_ms) for r in rows] == [
        ("VOICE", "xin chào", 420)
    ]
    assert rows[0].cv_features == {"color_name": "Red"}
