import asyncio
import base64
import binascii
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast

from pydantic import TypeAdapter, ValidationError

from app.schemas import contracts as c
from app.services.agent_tools import SqlStoreGateway, StoreGateway
from app.services.nlp_agent import (
    AgentDone,
    AgentEffects,
    AgentTextDelta,
    latest_cv_key,
    spatial_key,
)
from app.services.speech_service import Language, SpeechError, SpeechService
from app.services.vision_service import ImageDecodeError, VisionResult

logger = logging.getLogger("app.session")

AUDIO_FRAME_PREFIX = 0x01
MAX_AUDIO_FRAME_BYTES = 64 * 1024
MAX_UTTERANCE_BYTES = 30 * 16000 * 2  # 30 s of 16 kHz, 16-bit mono audio

MAX_CV_BYTES = 64 * 1024
MAX_CV_BASE64_CHARS = MAX_CV_BYTES * 4 // 3 + 8
CV_MAX_PER_SECOND = 2
QUALITY_WARNING_REPEAT_S = 5.0

LATEST_CV_TTL_S = 30
SPATIAL_TTL_S = 3600
LAST_WAYPOINT_TTL_S = 3600
ACTIVE_RESPONSE_TTL_S = 120

BARGE_IN_WAIT_S = 0.5
LATENCY_BUDGET_MS = 1500  # AUDIO_END -> first response chunk

DEFAULT_IDLE_TIMEOUT_S = 45.0
DEFAULT_MAX_CONNECTIONS_PER_USER = 3

CLIENT_MESSAGE_ADAPTER: TypeAdapter[Any] = TypeAdapter(c.ClientMessage)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…;])\s+|\n+")


def active_response_key(session_id: str) -> str:
    return f"session:{session_id}:active_response"


def last_waypoint_key(session_id: str) -> str:
    return f"session:{session_id}:last_waypoint"


def response_seq_key(session_id: str) -> str:
    return f"session:{session_id}:response_seq"


def now_ms() -> int:
    return int(time.time() * 1000)


async def _aw(value: Any) -> Any:
    return await cast("Awaitable[Any]", value)


def take_sentences(buffer: str) -> tuple[list[str], str]:
    """Split streamed text into finished sentences and the unfinished remainder."""
    parts = _SENTENCE_BOUNDARY.split(buffer)
    if len(parts) == 1:
        return [], buffer
    *complete, rest = parts
    return [p.strip() for p in complete if p.strip()], rest


# ---------------------------------------------------------------------------
# Backend (database) interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    user_id: Any
    session_id: str


class SessionBackend(Protocol):
    async def authenticate(
        self, session_id: str, token: str | None
    ) -> Principal | None:
        """The principal for a valid token that belongs to this session, else None."""
        ...

    async def cart_summary(self, user_id: Any) -> c.CartSummary: ...

    def gateway(self, user_id: Any) -> AbstractAsyncContextManager[StoreGateway]: ...

    async def log_interaction(
        self,
        session_id: str,
        modality: str,
        transcript: str | None,
        cv_features: dict[str, Any] | None,
        response: str,
        latency_ms: int | None,
    ) -> None: ...


class SqlBackend:
    """SessionBackend on PostgreSQL. Every operation uses its own database session."""

    def __init__(self, session_factory: Any) -> None:
        self._factory = session_factory

    async def authenticate(
        self, session_id: str, token: str | None
    ) -> Principal | None:
        if not token:
            return None
        import uuid

        from app.core.config import get_settings
        from app.models.domain import ChatSession
        from app.services import auth_service

        try:
            claims = auth_service.decode_access_token(token, get_settings())
            if claims.session_id != session_id:
                return None
            session_uuid = uuid.UUID(session_id)
        except (auth_service.AuthError, ValueError):
            return None
        async with self._factory() as db:
            try:
                await auth_service.load_user(db, claims)
            except auth_service.AuthError:
                return None
            chat_session = await db.get(ChatSession, session_uuid)
            if chat_session is None or chat_session.user_id != claims.user_id:
                return None
        return Principal(claims.user_id, session_id)

    async def cart_summary(self, user_id: Any) -> c.CartSummary:
        from app.services import cart_service

        async with self._factory() as db:
            return await cart_service.get_cart_summary(db, user_id)

    @asynccontextmanager
    async def gateway(self, user_id: Any) -> AsyncIterator[StoreGateway]:
        async with self._factory() as db:
            yield SqlStoreGateway(db, user_id)

    async def log_interaction(
        self,
        session_id: str,
        modality: str,
        transcript: str | None,
        cv_features: dict[str, Any] | None,
        response: str,
        latency_ms: int | None,
    ) -> None:
        import uuid

        from app.models.domain import InteractionLog

        async with self._factory() as db:
            db.add(
                InteractionLog(
                    session_id=uuid.UUID(session_id),
                    modality=modality,
                    user_transcript=transcript,
                    cv_features=cv_features,
                    agent_response=response,
                    latency_ms=latency_ms,
                )
            )
            await db.commit()


# ---------------------------------------------------------------------------
# One spoken utterance
# ---------------------------------------------------------------------------


class Utterance:
    """Audio frames of one spoken sentence, from the first frame to AUDIO_END."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._bytes = 0
        self._finished = False
        self.audio_end_at: float | None = None

    def push(self, pcm: bytes) -> bool:
        """Queue a frame; False once the utterance is too long (it is then finished)."""
        if self._finished:
            return False
        self._bytes += len(pcm)
        if self._bytes > MAX_UTTERANCE_BYTES:
            self.finish()
            return False
        self._queue.put_nowait(pcm)
        return True

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            self.audio_end_at = time.monotonic()
            self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


@dataclass
class _ChunkState:
    seq: int = 0
    animation_hint: str | None = None
    animation_sent: bool = False
    first_chunk_at: float | None = None
    tts_error_reported: bool = False


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

SendText = Callable[[str], Awaitable[None]]


class Session:
    def __init__(
        self, manager: "SessionManager", principal: Principal, send: SendText
    ) -> None:
        self._manager = manager
        self.principal = principal
        self.session_id = principal.session_id
        self._send_text = send
        self._send_lock = asyncio.Lock()
        self.closed = False

        self._tasks: set[asyncio.Task[Any]] = set()
        self._jobs: dict[str, asyncio.Task[Any]] = {}
        self._cancelled: set[str] = set()
        self.active_response_id: str | None = None
        self._utterance: Utterance | None = None

        self._cv_task: asyncio.Task[Any] | None = None
        self._cv_times: deque[float] = deque()
        self._warning_sent: tuple[str, float] | None = None

    # -- sending ------------------------------------------------------------

    async def send(self, message: Any, response_id: str | None = None) -> bool:
        """Send one message. A message tagged with a cancelled `response_id` is dropped
        (checked under the lock, so it is atomic with RESPONSE_CANCELLED)."""
        if self.closed:
            return False
        text = json.dumps(message.model_dump(mode="json"), ensure_ascii=False)
        async with self._send_lock:
            if self.closed or (response_id and response_id in self._cancelled):
                return False
            try:
                await self._send_text(text)
            except Exception:
                self.closed = True
                return False
        return True

    async def _send_error(
        self, code: c.WsErrorCode, message: str, recoverable: bool = True
    ) -> None:
        await self.send(
            c.WsErrorMessage(
                type="ERROR",
                timestamp=now_ms(),
                payload=c.WsErrorPayload(
                    code=code, message=message, recoverable=recoverable
                ),
            )
        )

    async def send_sync(self) -> None:
        redis = self._manager.redis
        cart = await self._manager.backend.cart_summary(self.principal.user_id)
        spatial, waypoint = await asyncio.gather(
            _aw(redis.hgetall(spatial_key(self.session_id))),
            _aw(redis.get(last_waypoint_key(self.session_id))),
        )

        def number(key: str) -> float:
            try:
                return float((spatial or {})[key])
            except (KeyError, TypeError, ValueError):
                return 0.0

        await self.send(
            c.SessionSyncMessage(
                type="SESSION_SYNC",
                timestamp=now_ms(),
                payload=c.SessionSyncPayload(
                    cart=cart,
                    spatial=c.SpatialState(
                        position=c.Vector3(
                            x=number("pos_x"), y=number("pos_y"), z=number("pos_z")
                        ),
                        current_aisle=(spatial or {}).get("current_aisle") or None,
                    ),
                    last_waypoint_id=waypoint or None,
                ),
            )
        )

    # -- task bookkeeping ---------------------------------------------------

    def _spawn(self, coro: Awaitable[Any], name: str) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "Session task %s failed", task.get_name(), exc_info=task.exception()
            )

    # -- incoming messages --------------------------------------------------

    async def on_text(self, raw: str) -> None:
        received_at = time.monotonic()
        try:
            message = CLIENT_MESSAGE_ADAPTER.validate_python(json.loads(raw))
        except (ValueError, ValidationError) as exc:
            logger.info("Invalid client message: %s", type(exc).__name__)
            await self._send_error("INTERNAL", "Invalid message.")
            return

        kind = message.type
        if kind == "HEARTBEAT":
            await self.send(
                c.HeartbeatMessage(
                    type="HEARTBEAT", timestamp=now_ms(), payload=c.EmptyPayload()
                )
            )
        elif kind == "RESYNC_REQUEST":
            await self.send_sync()
        elif kind == "PLAYER_TELEMETRY":
            await self._on_telemetry(message.payload)
        elif kind == "CV_FRAME_ROI":
            self._on_cv_frame(message.payload)
        elif kind == "AUDIO_END":
            self._on_audio_end()
        elif kind == "BARGE_IN_INTERRUPT":
            await self.handle_barge_in(message.payload.response_id)
        elif kind == "TEXT_MESSAGE":
            payload = message.payload
            self._spawn(
                self._respond(
                    payload.text,
                    payload.language,
                    want_audio=payload.want_audio,
                    modality="TEXT",
                    started_at=received_at,
                ),
                "text-turn",
            )

    def on_audio_frame(self, data: bytes) -> None:
        """A binary frame: 0x01 followed by 16 kHz, 16-bit, mono PCM."""
        if len(data) < 2 or data[0] != AUDIO_FRAME_PREFIX:
            return
        pcm = data[1:]
        if len(pcm) % 2 or len(pcm) > MAX_AUDIO_FRAME_BYTES:
            return
        utterance = self._utterance
        if utterance is None:
            utterance = Utterance()
            self._utterance = utterance
            self._spawn(self._voice_turn(utterance), "voice-turn")
        if not utterance.push(pcm):
            self._utterance = None

    def _on_audio_end(self) -> None:
        utterance, self._utterance = self._utterance, None
        if utterance is not None:
            utterance.finish()

    # -- telemetry and camera -----------------------------------------------

    async def _on_telemetry(self, payload: c.PlayerMovedPayload) -> None:
        key = spatial_key(self.session_id)
        mapping = {
            "pos_x": str(payload.position.x),
            "pos_y": str(payload.position.y),
            "pos_z": str(payload.position.z),
            "yaw": str(payload.yaw),
            "pitch": str(payload.pitch),
            "current_aisle": payload.current_aisle or "",
            "gaze_product_id": payload.gaze_target_product_id or "",
            "updated_at": str(now_ms()),
        }
        redis = self._manager.redis
        try:
            await _aw(redis.hset(key, mapping=mapping))
            await _aw(redis.expire(key, SPATIAL_TTL_S))
        except Exception:
            logger.warning("Could not store telemetry", exc_info=True)

    def _on_cv_frame(self, payload: c.CvFrameRoiPayload) -> None:
        if len(payload.image_base64) > MAX_CV_BASE64_CHARS:
            logger.info("Dropped CV frame: larger than %d bytes", MAX_CV_BYTES)
            return
        now = time.monotonic()
        while self._cv_times and now - self._cv_times[0] >= 1.0:
            self._cv_times.popleft()
        if len(self._cv_times) >= CV_MAX_PER_SECOND:
            return  # more than 2 frames per second
        if self._cv_task is not None and not self._cv_task.done():
            return  # still analysing the previous frame
        self._cv_times.append(now)
        self._cv_task = self._spawn(self._process_cv(payload), "cv-frame")

    async def _process_cv(self, payload: c.CvFrameRoiPayload) -> None:
        try:
            image = base64.b64decode(payload.image_base64, validate=True)
        except (binascii.Error, ValueError):
            await self._send_error(
                "CV_UNSUPPORTED_IMAGE", "The image is not valid base64."
            )
            return
        if len(image) > MAX_CV_BYTES:
            return
        try:
            result: VisionResult = await self._manager.vision.analyze(image)
        except ImageDecodeError:
            await self._send_error(
                "CV_UNSUPPORTED_IMAGE", "The image could not be read."
            )
            return
        except Exception:
            logger.exception("Vision analysis failed")
            await self._send_error("INTERNAL", "Image analysis is unavailable.")
            return

        redis = self._manager.redis
        features = {
            "lab": list(result.lab),
            "color_name": result.color_name,
            "category_label": result.category_label,
            "category_confidence": result.category_confidence,
            "quality_warning": result.quality_warning,
            "embedding": result.embedding,
            "captured_at": now_ms(),
        }
        try:
            await _aw(
                redis.set(
                    latest_cv_key(self.session_id),
                    json.dumps(features),
                    ex=LATEST_CV_TTL_S,
                )
            )
            if payload.gaze_target_product_id:
                key = spatial_key(self.session_id)
                gaze = {"gaze_product_id": payload.gaze_target_product_id}
                await _aw(redis.hset(key, mapping=gaze))
                await _aw(redis.expire(key, SPATIAL_TTL_S))
        except Exception:
            logger.warning("Could not store camera features", exc_info=True)
        await self._report_quality(result)

    async def _report_quality(self, result: VisionResult) -> None:
        warning = result.quality_warning
        if warning is None:
            self._warning_sent = None
            return
        now = time.monotonic()
        previous = self._warning_sent
        if previous is not None and previous[0] == warning:
            if now - previous[1] < QUALITY_WARNING_REPEAT_S:
                return
        self._warning_sent = (warning, now)
        await self.send(
            c.CvQualityWarningMessage(
                type="CV_QUALITY_WARNING",
                timestamp=now_ms(),
                payload=c.CvQualityWarningPayload(
                    warning=warning,
                    mean_luminance=max(0.0, min(255.0, result.mean_luminance)),
                ),
            )
        )

    # -- voice and text turns -----------------------------------------------

    async def _voice_turn(self, utterance: Utterance) -> None:
        final = None
        try:
            async for result in self._manager.speech.transcribe(utterance.frames()):
                await self.send(
                    c.SttTranscriptMessage(
                        type="STT_TRANSCRIPT",
                        timestamp=now_ms(),
                        payload=c.SttTranscriptPayload(
                            text=result.text,
                            confidence=max(0.0, min(1.0, result.confidence)),
                            is_final=result.is_final,
                            language=result.language,
                        ),
                    )
                )
                if result.is_final:
                    final = result
        except SpeechError as exc:
            code: c.WsErrorCode = (
                "STT_FAILED" if exc.code == "STT_FAILED"
                else "RATE_LIMITED" if exc.code == "RATE_LIMITED"
                else "INTERNAL"
            )
            await self._send_error(code, exc.message, exc.recoverable)
            return
        if final is None or not final.text.strip():
            await self._send_error("STT_FAILED", "No speech was recognised.")
            return
        await self._respond(
            final.text,
            final.language,
            want_audio=True,
            modality="VOICE",
            started_at=utterance.audio_end_at or time.monotonic(),
        )

    async def _new_response_id(self) -> str:
        redis = self._manager.redis
        key = response_seq_key(self.session_id)
        number = int(await _aw(redis.incr(key)))
        await _aw(redis.expire(key, SPATIAL_TTL_S))
        return f"resp_{number:04d}"

    async def _respond(
        self,
        text: str,
        language: Language,
        *,
        want_audio: bool,
        modality: str,
        started_at: float,
    ) -> None:
        if self.active_response_id is not None:
            await self.handle_barge_in(self.active_response_id)
        response_id = await self._new_response_id()
        task = asyncio.current_task()
        if task is not None:
            self._jobs[response_id] = task
        self.active_response_id = response_id
        redis = self._manager.redis
        try:
            active_key = active_response_key(self.session_id)
            await _aw(redis.set(active_key, response_id, ex=ACTIVE_RESPONSE_TTL_S))
            await self._stream_response(
                response_id, text, language, want_audio, modality, started_at
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Response %s failed", response_id)
            await self._send_error("INTERNAL", "Something went wrong while answering.")
        finally:
            self._jobs.pop(response_id, None)
            if self.active_response_id == response_id:
                self.active_response_id = None
                try:
                    await _aw(redis.delete(active_response_key(self.session_id)))
                except Exception:
                    logger.warning("Could not clear active_response", exc_info=True)

    async def handle_barge_in(self, response_id: str) -> None:
        """Cancel a response and confirm with RESPONSE_CANCELLED."""
        self._cancelled.add(response_id)  # from now on none of its chunks are sent
        task = self._jobs.get(response_id)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=BARGE_IN_WAIT_S)
        if self.active_response_id == response_id:
            self.active_response_id = None
            try:
                active_key = active_response_key(self.session_id)
                await _aw(self._manager.redis.delete(active_key))
            except Exception:
                logger.warning("Could not clear active_response", exc_info=True)
        await self.send(
            c.ResponseCancelledMessage(
                type="RESPONSE_CANCELLED",
                timestamp=now_ms(),
                payload=c.ResponseCancelledPayload(response_id=response_id),
            )
        )

    async def _stream_response(
        self,
        response_id: str,
        user_text: str,
        language: Language,
        want_audio: bool,
        modality: str,
        started_at: float,
    ) -> None:
        manager = self._manager
        state = _ChunkState()
        buffer = ""
        done: AgentDone | None = None

        async with manager.backend.gateway(self.principal.user_id) as gateway:
            events = manager.agent().respond(
                gateway=gateway,
                redis=manager.redis,
                session_id=self.session_id,
                user_text=user_text,
                language=language,
            )
            try:
                async for event in events:
                    if isinstance(event, AgentTextDelta):
                        buffer += event.text
                        sentences, buffer = take_sentences(buffer)
                        for sentence in sentences:
                            await self._emit_text(
                                response_id, state, sentence, language, want_audio
                            )
                    elif isinstance(event, AgentEffects):
                        await self._emit_effects(response_id, state, event)
                    elif isinstance(event, AgentDone):
                        done = event
            finally:
                closer = getattr(events, "aclose", None)
                if closer is not None:
                    await closer()

        products = list(done.recommended_products) if done else []
        animation = done.avatar_animation if done else None
        await self._emit_text(
            response_id,
            state,
            buffer.strip(),
            language,
            want_audio,
            final=True,
            products=products,
            animation=animation,
        )

        reply = done.text if done else buffer.strip()
        latency_ms: int | None = None
        if state.first_chunk_at is not None:
            latency_ms = round((state.first_chunk_at - started_at) * 1000)
            manager.record_latency(latency_ms)
            log = logger.warning if latency_ms > LATENCY_BUDGET_MS else logger.info
            log(
                "response %s: %d ms from %s to first chunk (budget %d ms)",
                response_id,
                latency_ms,
                "AUDIO_END" if modality == "VOICE" else "message",
                LATENCY_BUDGET_MS,
            )
        await self._log(modality, user_text, reply, latency_ms)

    async def _log(
        self, modality: str, transcript: str, reply: str, latency_ms: int | None
    ) -> None:
        features: dict[str, Any] | None = None
        try:
            raw = await _aw(self._manager.redis.get(latest_cv_key(self.session_id)))
            if raw:
                data = json.loads(raw)
                features = {
                    key: data.get(key)
                    for key in ("color_name", "category_label", "category_confidence")
                }
            await self._manager.backend.log_interaction(
                self.session_id, modality, transcript, features, reply, latency_ms
            )
        except Exception:
            logger.warning("Could not log the interaction", exc_info=True)

    # -- emitting -----------------------------------------------------------

    async def _emit_effects(
        self, response_id: str, state: _ChunkState, effects: AgentEffects
    ) -> None:
        if effects.commands:
            for command in effects.commands:
                is_animation = command.action == "PLAY_AVATAR_ANIMATION"
                if is_animation and not state.animation_hint:
                    state.animation_hint = command.payload.animation
                if command.action == "NAVIGATE_TO_WAYPOINT":
                    await self._remember_waypoint(command.payload.waypoint_id)
            await self.send(
                c.UnityCommandMessage(
                    type="UNITY_COMMAND",
                    timestamp=now_ms(),
                    payload=c.UnityCommandPayload(
                        response_id=response_id, commands=list(effects.commands)
                    ),
                ),
                response_id,
            )
        if effects.ui_action is not None:
            await self.send(
                c.UiActionMessage(
                    type="UI_ACTION", timestamp=now_ms(), payload=effects.ui_action
                ),
                response_id,
            )
        if effects.cart is not None:
            await self.send(
                c.CartUpdatedMessage(
                    type="CART_UPDATED", timestamp=now_ms(), payload=effects.cart
                ),
                response_id,
            )

    async def _remember_waypoint(self, waypoint_id: str) -> None:
        try:
            key = last_waypoint_key(self.session_id)
            await _aw(self._manager.redis.set(key, waypoint_id, ex=LAST_WAYPOINT_TTL_S))
        except Exception:
            logger.warning("Could not store last_waypoint", exc_info=True)

    async def _emit_text(
        self,
        response_id: str,
        state: _ChunkState,
        text: str,
        language: Language,
        want_audio: bool,
        *,
        final: bool = False,
        products: list[str] | None = None,
        animation: str | None = None,
    ) -> None:
        """Send `text` as one or more chunks.

        With `final` the last chunk ends the response (an empty chunk is sent when
        there is no text left).
        """
        parts: list[Any] = []  # SpeechChunk, or None for a text-only chunk
        if text and want_audio:
            try:
                async for chunk in self._manager.speech.speak(text, language):
                    if final:
                        parts.append(chunk)
                    else:
                        await self._send_chunk(
                            response_id, state, text=chunk.text, speech=chunk
                        )
            except SpeechError as exc:
                logger.warning("TTS failed (%s); sending text only", exc.code)
                if not state.tts_error_reported:
                    state.tts_error_reported = True
                    code = exc.code if exc.code in ("RATE_LIMITED",) else "INTERNAL"
                    await self._send_error(code, exc.message)
                if final:
                    parts = [None]
                else:
                    await self._send_chunk(response_id, state, text=text)
            if not final:
                return
        elif text:
            if not final:
                await self._send_chunk(response_id, state, text=text)
                return
            parts = [None]

        if not final:
            return
        if not parts:
            await self._send_chunk(
                response_id,
                state,
                text="",
                final=True,
                products=products,
                animation=animation,
            )
            return
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            await self._send_chunk(
                response_id,
                state,
                text=part.text if part is not None else text,
                speech=part,
                final=last,
                products=products if last else None,
                animation=animation,
            )

    async def _send_chunk(
        self,
        response_id: str,
        state: _ChunkState,
        *,
        text: str,
        speech: Any = None,
        final: bool = False,
        products: list[str] | None = None,
        animation: str | None = None,
    ) -> None:
        avatar_animation: str | None = None
        if not state.animation_sent:
            avatar_animation = state.animation_hint or (animation if final else None)
            if avatar_animation:
                state.animation_sent = True

        message = c.AgentResponseStreamMessage(
            type="AGENT_RESPONSE_STREAM",
            timestamp=now_ms(),
            payload=c.AgentResponseChunk(
                response_id=response_id,
                chunk_seq=state.seq,
                is_final=final,
                text=text,
                audio_base64=(
                    base64.b64encode(speech.audio).decode("ascii") if speech else None
                ),
                audio_format=speech.audio_format if speech else None,
                audio_duration_ms=speech.duration_ms if speech else 0,
                visemes=speech.viseme_frames() if speech else [],
                recommended_products=products if final and products else None,
                avatar_animation=cast(Any, avatar_animation),
            ),
        )
        if await self.send(message, response_id):
            state.seq += 1
            if state.first_chunk_at is None:
                state.first_chunk_at = time.monotonic()

    # -- shutdown -----------------------------------------------------------

    async def close(self) -> None:
        self.closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=1.0)
        try:
            await _aw(self._manager.redis.delete(active_response_key(self.session_id)))
        except Exception:
            logger.warning("Could not clear active_response", exc_info=True)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Shared services plus per-user connection limits."""

    def __init__(
        self,
        *,
        redis: Any,
        backend: SessionBackend,
        vision: Any,
        speech: SpeechService,
        agent_provider: Callable[[], Any],
        max_connections_per_user: int = DEFAULT_MAX_CONNECTIONS_PER_USER,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        allowed_origins: tuple[str, ...] = (),
    ) -> None:
        self.redis = redis
        self.backend = backend
        self.vision = vision
        self.speech = speech
        self._agent_provider = agent_provider
        self._agent: Any = None
        self.max_connections_per_user = max_connections_per_user
        self.idle_timeout_s = idle_timeout_s
        self.allowed_origins = allowed_origins
        self._connections: dict[Any, int] = {}
        self.latencies_ms: deque[int] = deque(maxlen=1000)

    def agent(self) -> Any:
        if self._agent is None:
            self._agent = self._agent_provider()
        return self._agent

    def origin_allowed(self, origin: str | None) -> bool:
        return not self.allowed_origins or origin in self.allowed_origins

    def try_reserve(self, user_id: Any) -> bool:
        count = self._connections.get(user_id, 0)
        if count >= self.max_connections_per_user:
            return False
        self._connections[user_id] = count + 1
        return True

    def release(self, user_id: Any) -> None:
        count = self._connections.get(user_id, 0)
        if count <= 1:
            self._connections.pop(user_id, None)
        else:
            self._connections[user_id] = count - 1

    def create_session(self, principal: Principal, send: SendText) -> Session:
        return Session(self, principal, send)

    def record_latency(self, latency_ms: int) -> None:
        self.latencies_ms.append(latency_ms)

    async def warm_up(self) -> None:
        """Load the vision model before the first request (failure is logged only)."""
        load = getattr(self.vision, "load", None)
        if load is None:
            return
        try:
            await load()
        except Exception:
            logger.exception("Vision model could not be loaded; camera search is off")


def create_default_session_manager(redis: Any, session_factory: Any) -> SessionManager:
    """The production manager: SQL backend, real speech/vision/agent services.

    Environment: WS_MAX_CONNECTIONS_PER_USER (default 3) and ALLOWED_ORIGINS
    (comma-separated; empty allows any origin).
    """
    from app.services.nlp_agent import get_shopkeeper_agent
    from app.services.speech_service import get_speech_service
    from app.services.vision_service import get_vision_service

    origins = tuple(
        o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
    )
    return SessionManager(
        redis=redis,
        backend=SqlBackend(session_factory),
        vision=get_vision_service(),
        speech=get_speech_service(),
        agent_provider=get_shopkeeper_agent,
        max_connections_per_user=int(
            os.environ.get(
                "WS_MAX_CONNECTIONS_PER_USER", DEFAULT_MAX_CONNECTIONS_PER_USER
            )
        ),
        allowed_origins=origins,
    )
