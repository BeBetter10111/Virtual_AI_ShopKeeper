import asyncio
import json
import os
import random
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pytest
import pytest_asyncio
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.models.domain import User
from app.schemas import contracts as c
from app.services import nlp_agent as agent_module
from app.services.agent_tools import (
    CategoryInfo,
    ProductInfo,
    RankedProduct,
    SpatialContext,
    SqlStoreGateway,
    ToolContext,
    VariantInfo,
    VisionContext,
    run_tool,
)
from app.services.cart_service import CartError
from app.services.nlp_agent import (
    AgentDone,
    AgentEffects,
    AgentTextDelta,
    ShopkeeperAgent,
    WordBudget,
)
from app.services.vision_service import delta_e_2000, lab_from_hex
from scripts.seed_store import seed

CANCEL_BUDGET_S = 0.1


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class Chunk:
    """A streamed model chunk shaped like LangChain's AIMessageChunk."""

    def __init__(
        self,
        content: Any = "",
        tool_call_chunks: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []
        self.tool_calls = tool_calls or []


def text_chunks(sentence: str) -> list[Chunk]:
    return [Chunk(word + " ") for word in sentence.split()]


def call_chunks(
    name: str, args: dict[str, Any] | str, call_id: str = "call_1", index: int = 0
) -> list[Chunk]:
    """A tool call streamed the way providers do it: name first, JSON args in pieces."""
    raw = args if isinstance(args, str) else json.dumps(args)
    third = max(1, len(raw) // 3)
    pieces = [raw[:third], raw[third : 2 * third], raw[2 * third :]]
    head = {"name": name, "args": "", "id": call_id, "index": index}
    chunks = [Chunk(tool_call_chunks=[head])]
    chunks += [
        Chunk(tool_call_chunks=[{"name": None, "args": p, "id": None, "index": index}])
        for p in pieces
        if p
    ]
    return chunks


class FakeModel:
    """Scripted chat model. Each call to astream plays the next scripted turn."""

    def __init__(self, turns: list[list[Chunk]], delay: float = 0.0) -> None:
        self.turns = list(turns)
        self.delay = delay
        self.calls: list[tuple[bool, list[dict[str, Any]]]] = []
        self.closed = 0
        self.bound_schemas: list[dict[str, Any]] | None = None

    def bind_tools(self, tools: list[dict[str, Any]]) -> "FakeModel._Bound":
        self.bound_schemas = tools
        return FakeModel._Bound(self)

    class _Bound:
        def __init__(self, parent: "FakeModel") -> None:
            self._parent = parent

        def astream(
            self, messages: list[dict[str, Any]]
        ) -> AsyncGenerator[Chunk, None]:
            return self._parent._play(messages, with_tools=True)

    def astream(self, messages: list[dict[str, Any]]) -> AsyncGenerator[Chunk, None]:
        return self._play(messages, with_tools=False)

    async def _play(
        self, messages: list[dict[str, Any]], with_tools: bool
    ) -> AsyncGenerator[Chunk, None]:
        self.calls.append((with_tools, [dict(m) for m in messages]))
        turn = self.turns.pop(0) if self.turns else []
        try:
            for chunk in turn:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield chunk
        finally:
            self.closed += 1


class FakeRedis:
    def __init__(
        self,
        spatial: dict[str, str] | None = None,
        latest_cv: str | None = None,
        memory: list[str] | None = None,
    ) -> None:
        self.hashes = {"session:s1:spatial": dict(spatial or {})}
        self.values = {"session:s1:latest_cv": latest_cv} if latest_cv else {}
        self.lists = {"session:s1:memory": list(memory or [])}
        self.ttls: dict[str, int] = {}

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        items = self.lists.get(key, [])
        stop = len(items) if stop == -1 else stop + 1
        start = max(len(items) + start, 0) if start < 0 else start
        return items[start:stop]

    async def rpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    async def ltrim(self, key: str, start: int, stop: int) -> bool:
        items = self.lists.get(key, [])
        start = max(len(items) + start, 0) if start < 0 else start
        self.lists[key] = items[start:] if stop == -1 else items[start : stop + 1]
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        self.ttls[key] = seconds
        return True


def variants(
    product_number: int, sizes: Sequence[str], stock: Sequence[int], color: str
) -> tuple[VariantInfo, ...]:
    return tuple(
        VariantInfo(f"var_{product_number:03d}_{size}", size, color, qty, 0)
        for size, qty in zip(sizes, stock, strict=True)
    )


def make_product(
    number: int, name: str, price: int, slug: str, hex_color: str, color: str,
    sizes: Sequence[str], stock: Sequence[int], waypoint: str, aisle: str,
) -> ProductInfo:
    return ProductInfo(
        product_id=f"prod_{number:03d}",
        name=name,
        description=f"{name} description",
        base_price=price,
        category_slug=slug,
        category_name=slug.capitalize(),
        waypoint_id=waypoint,
        aisle_code=aisle,
        color_lab=lab_from_hex(hex_color),
        variants=variants(number, sizes, stock, color),
    )


SHIRT_SIZES = ("S", "M", "L", "XL")
CATALOGUE = [
    make_product(1, "Blue Oxford Shirt", 450000, "shirts", "#1F3A5F", "Navy Blue",
                 SHIRT_SIZES, (8, 5, 0, 3), "WP_AISLE_01_SHIRTS", "A1"),
    make_product(2, "White Linen Shirt", 520000, "shirts", "#F2F2EE", "White",
                 SHIRT_SIZES, (6, 9, 7, 2), "WP_AISLE_01_SHIRTS", "A1"),
    make_product(3, "Red Flannel Shirt", 480000, "shirts", "#B22222", "Red",
                 SHIRT_SIZES, (0, 0, 0, 0), "WP_AISLE_01_SHIRTS", "A1"),
    make_product(4, "Brick Red Cotton Shirt", 430000, "shirts", "#A83232", "Brick Red",
                 SHIRT_SIZES, (4, 6, 5, 3), "WP_AISLE_01_SHIRTS", "A1"),
    make_product(5, "Black Leather Jacket", 2490000, "jackets", "#1A1A1A", "Black",
                 SHIRT_SIZES, (2, 3, 2, 1), "WP_AISLE_02_JACKETS", "A2"),
    make_product(11, "White Sneakers", 1250000, "shoes", "#F5F5F5", "White",
                 ("40", "41", "42", "43"), (4, 6, 5, 2), "WP_AISLE_04_SHOES", "A4"),
]
CATEGORIES = [
    CategoryInfo("jackets", "Jackets", "WP_AISLE_02_JACKETS", "A2"),
    CategoryInfo("pants", "Pants", "WP_AISLE_03_PANTS", "A3"),
    CategoryInfo("shirts", "Shirts", "WP_AISLE_01_SHIRTS", "A1"),
    CategoryInfo("shoes", "Shoes", "WP_AISLE_04_SHOES", "A4"),
]


def product_of(variant_id: str) -> ProductInfo:
    prefix_of = {p: f"var_{p.product_id[5:]}_" for p in CATALOGUE}
    return next(p for p, prefix in prefix_of.items() if variant_id.startswith(prefix))


class FakeGateway:
    """In-memory StoreGateway mirroring the seed catalogue."""

    def __init__(self, fail_on_get: bool = False) -> None:
        self.fail_on_get = fail_on_get
        self.search_calls: list[dict[str, Any]] = []
        self.cart_lines: dict[str, int] = {}
        self.cart_version = 0

    async def get_product(self, product_id: str) -> ProductInfo | None:
        if self.fail_on_get:
            raise RuntimeError("database is down")
        return next((p for p in CATALOGUE if p.product_id == product_id), None)

    async def list_categories(self) -> list[CategoryInfo]:
        return list(CATEGORIES)

    async def search(
        self, *, query_text: str | None, lab: Sequence[float] | None,
        embedding: Sequence[float] | None, category_slug: str | None, limit: int,
    ) -> list[RankedProduct]:
        self.search_calls.append(
            {"query_text": query_text, "lab": lab, "embedding": embedding,
             "category_slug": category_slug}
        )
        words = (query_text or "").lower().split()
        found = [
            p for p in CATALOGUE
            if (category_slug is None or p.category_slug == category_slug)
            and all(w in p.name.lower() for w in words)
        ]
        ranked = []
        for p in found:
            delta = delta_e_2000(lab, p.color_lab) if lab and p.color_lab else None
            score = max(0.0, 1 - delta / 50) if delta is not None else 0.0
            ranked.append(RankedProduct(p, delta, score))
        ranked.sort(key=lambda r: (-r.score, r.product.product_id))
        return ranked[:limit]

    async def alternatives_for(
        self, product_id: str, lab: Sequence[float] | None
    ) -> list[RankedProduct]:
        product = await self.get_product(product_id)
        assert product is not None
        reference = lab if lab is not None else product.color_lab
        near = []
        for p in CATALOGUE:
            if p.category_slug != product.category_slug or p.product_id == product_id:
                continue
            delta = delta_e_2000(reference, p.color_lab)
            if p.in_stock and delta < 10:
                near.append(RankedProduct(p, delta, 1 - delta / 50))
        near.sort(key=lambda r: r.delta_e00 or 0.0)
        return near[:2]

    def _summary(self) -> c.CartSummary:
        items = []
        for variant_id, quantity in sorted(self.cart_lines.items()):
            product = product_of(variant_id)
            variant = next(v for v in product.variants if v.variant_id == variant_id)
            items.append(
                c.CartItem(
                    product_id=product.product_id,
                    variant_id=variant_id,
                    name=product.name,
                    size=variant.size,
                    quantity=quantity,
                    unit_price=product.base_price,
                )
            )
        return c.CartSummary(
            cart_id="cart-1", version=self.cart_version, items=items,
            item_count=sum(i.quantity for i in items),
            total_amount=sum(i.quantity * i.unit_price for i in items),
        )

    async def add_to_cart(self, variant_id: str, quantity: int) -> c.CartSummary:
        product = product_of(variant_id)
        variant = next(v for v in product.variants if v.variant_id == variant_id)
        wanted = self.cart_lines.get(variant_id, 0) + quantity
        if wanted > variant.stock:
            raise CartError(
                "STOCK_INSUFFICIENT", 409, "no", {"available": variant.stock}
            )
        self.cart_lines[variant_id] = wanted
        self.cart_version += 1
        return self._summary()

    async def cart_summary(self) -> c.CartSummary:
        return self._summary()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def vision_json(
    hex_color: str = "#B22222",
    label: str = "shirt",
    confidence: float = 0.9,
    warning: str | None = None,
) -> str:
    lab = lab_from_hex(hex_color)
    return json.dumps(
        {"lab": list(lab), "color_name": "Red", "category_label": label,
         "category_confidence": confidence, "quality_warning": warning,
         "embedding": [0.1] * 512}
    )


async def run_agent(
    model: FakeModel,
    *,
    user: str = "xin chào",
    language: str = "vi",
    gateway: FakeGateway | None = None,
    redis: FakeRedis | None = None,
    **agent_kwargs: Any,
) -> list[Any]:
    agent = ShopkeeperAgent(model, **agent_kwargs)
    events = []
    async for event in agent.respond(
        gateway=gateway or FakeGateway(), redis=redis or FakeRedis(),
        session_id="s1", user_text=user, language=language,  # type: ignore[arg-type]
    ):
        events.append(event)
    return events


def spoken(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, AgentTextDelta)).strip()


def effects(events: list[Any]) -> list[AgentEffects]:
    return [e for e in events if isinstance(e, AgentEffects)]


def done(events: list[Any]) -> AgentDone:
    assert isinstance(events[-1], AgentDone)
    return events[-1]


def command_names(effect: AgentEffects) -> list[str]:
    return [cmd.action for cmd in effect.commands]


def validate_commands(effect: AgentEffects) -> None:
    adapter = TypeAdapter(c.UnityCommand)
    for command in effect.commands:
        adapter.validate_python(command.model_dump())


# ---------------------------------------------------------------------------
# WordBudget
# ---------------------------------------------------------------------------


def test_word_budget_passes_short_text_through() -> None:
    budget = WordBudget(40)
    assert budget.feed("Xin chào ") + budget.feed("bạn nhé.") == "Xin chào bạn nhé."
    assert budget.finish() == ""


def test_word_budget_cuts_at_exactly_the_limit() -> None:
    budget = WordBudget(5)
    out = "".join(budget.feed(w + " ") for w in "một hai ba bốn năm sáu bảy".split())
    assert out.split() == ["một", "hai", "ba", "bốn", "năm"]
    assert budget.exhausted is True
    assert budget.feed("thêm nữa") == ""


def test_word_budget_closes_a_cut_sentence() -> None:
    budget = WordBudget(3)
    budget.feed("one two three four five")
    assert budget.finish() == "."
    finished = WordBudget(2)
    assert finished.feed("one two. Three four") == "one two."
    assert finished.finish() == ""  # already ends at a full stop


def test_word_budget_handles_words_split_across_chunks() -> None:
    budget = WordBudget(2)
    pieces = ["ab", "c d", "ef gh", "i"]
    assert "".join(budget.feed(p) for p in pieces).split() == ["abc", "def"]


def test_word_budget_property_over_random_splits() -> None:
    rng = random.Random(7)
    words = [f"w{i}" for i in range(120)]
    text = " ".join(words)
    for _ in range(50):
        limit = rng.randint(1, 60)
        budget = WordBudget(limit)
        out, i = [], 0
        while i < len(text):
            step = rng.randint(1, 9)
            out.append(budget.feed(text[i : i + step]))
            i += step
        assert "".join(out).split() == words[:limit]


def test_vietnamese_words_are_counted_per_syllable() -> None:
    budget = WordBudget(4)
    words = budget.feed("Tôi muốn mua một chiếc áo").split()
    assert words == ["Tôi", "muốn", "mua", "một"]


# ---------------------------------------------------------------------------
# Context and prompt
# ---------------------------------------------------------------------------


def test_malformed_context_is_ignored() -> None:
    assert agent_module.parse_vision("not json") is None
    assert agent_module.parse_vision(json.dumps({"lab": [1, 2]})) is None
    assert agent_module.parse_vision(None) is None
    spatial = agent_module.parse_spatial({"pos_x": "oops", "gaze_product_id": ""})
    assert spatial.position is None and spatial.gaze_product_id is None
    messages, last = agent_module.parse_history(["junk", json.dumps({"user": "a"})])
    assert messages == [] and last is None


def test_history_becomes_messages_and_tracks_the_last_product() -> None:
    items = [
        json.dumps({"user": "u1", "assistant": "a1", "products": []}),
        json.dumps(
            {"user": "u2", "assistant": "a2", "products": ["prod_002", "prod_003"]}
        ),
    ]
    messages, last = agent_module.parse_history(items)
    assert [m["content"] for m in messages] == ["u1", "a1", "u2", "a2"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert last == "prod_002"


def test_system_prompt_mentions_gaze_camera_and_language() -> None:
    ctx = ToolContext(
        language="vi",
        spatial=SpatialContext(current_aisle="A1", gaze_product_id="prod_002"),
        vision=VisionContext((50.0, 0.0, 0.0), "Grey", "mug", 0.03, "LOW_LIGHT"),
    )
    prompt = agent_module.build_system_prompt(ctx, "White Linen Shirt", None)
    assert "Vietnamese" in prompt
    assert "aisle A1" in prompt
    assert "White Linen Shirt" in prompt and "prod_002" in prompt
    assert "does NOT sell" in prompt and "mug" in prompt
    assert "too dark" in prompt


def test_system_prompt_for_a_store_item_and_for_no_camera() -> None:
    ctx = ToolContext(
        language="en",
        vision=VisionContext((30.0, 5.0, -20.0), "Navy Blue", "shirt", 0.9),
        last_product_id="prod_001",
    )
    prompt = agent_module.build_system_prompt(ctx, None, "Blue Oxford Shirt")
    assert "English" in prompt and "sells" in prompt
    assert "discussed last: Blue Oxford Shirt" in prompt
    empty = agent_module.build_system_prompt(ToolContext(language="en"), None, None)
    assert "nothing is being shown" in empty
    assert "No product is being looked at" in empty


def test_chunk_text_handles_strings_and_content_blocks() -> None:
    assert agent_module.chunk_text(Chunk("abc")) == "abc"
    blocks = [{"type": "text", "text": "he"}, {"type": "tool_use", "id": "x"}, "llo"]
    assert agent_module.chunk_text(Chunk(blocks)) == "hello"
    assert agent_module.chunk_text(Chunk(None)) == ""


def test_default_model_kwargs_turn_reasoning_off() -> None:
    anthropic = agent_module.default_model_kwargs("anthropic", "claude-haiku-5-5")
    assert anthropic["thinking"] == {"type": "disabled"}
    flash = agent_module.default_model_kwargs("google_genai", "gemini-3.5-flash")
    assert flash["thinking_level"] == "minimal"
    lite = agent_module.default_model_kwargs("google_genai", "gemini-3.5-flash-lite")
    assert "thinking_level" not in lite
    assert all(k["timeout"] <= 10 for k in (anthropic, flash, lite))


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------


async def test_plain_answer_streams_and_is_remembered() -> None:
    model = FakeModel([text_chunks("Chào bạn, mình giúp gì được ạ?")])
    redis = FakeRedis()
    events = await run_agent(model, user="Xin chào", redis=redis)

    assert spoken(events) == "Chào bạn, mình giúp gì được ạ?"
    assert len(model.calls) == 1
    assert model.calls[0][0] is True  # tools were available
    assert effects(events) == []
    assert done(events).recommended_products == []
    entry = json.loads(redis.lists["session:s1:memory"][0])
    assert entry["user"] == "Xin chào"
    assert entry["assistant"] == "Chào bạn, mình giúp gì được ạ?"
    assert redis.ttls["session:s1:memory"] == agent_module.MEMORY_TTL_SECONDS


async def test_text_is_forwarded_as_it_streams() -> None:
    model = FakeModel([text_chunks("một hai ba bốn")], delay=0.05)
    agent = ShopkeeperAgent(model)
    started = time.perf_counter()
    first_at = None
    async for event in agent.respond(
        gateway=FakeGateway(), redis=FakeRedis(), session_id="s1",
        user_text="hi", language="vi",
    ):
        if isinstance(event, AgentTextDelta) and first_at is None:
            first_at = time.perf_counter() - started
    total = time.perf_counter() - started
    assert first_at is not None and first_at < total / 2


async def test_reply_is_capped_at_40_words() -> None:
    long_answer = " ".join(f"từ{i}" for i in range(80))
    events = await run_agent(FakeModel([text_chunks(long_answer)]))
    reply = spoken(events)
    assert len(reply.split()) == 40
    assert reply.endswith(".")
    assert done(events).text == reply


async def test_empty_model_output_gets_a_fallback_reply() -> None:
    vi = await run_agent(FakeModel([[]]), language="vi")
    en = await run_agent(FakeModel([[]]), language="en")
    assert "nói lại" in spoken(vi)
    assert "say it again" in spoken(en)


async def test_cancelling_stops_the_stream_within_100_ms_and_stores_nothing() -> None:
    model = FakeModel([text_chunks("một hai ba bốn năm sáu bảy tám")], delay=0.3)
    redis = FakeRedis()
    agent = ShopkeeperAgent(model)
    first = asyncio.Event()

    async def consume() -> None:
        async for event in agent.respond(
            gateway=FakeGateway(),
            redis=redis,
            session_id="s1",
            user_text="hi",
            language="vi",
        ):
            if isinstance(event, AgentTextDelta):
                first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), timeout=2)
    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.perf_counter() - started < CANCEL_BUDGET_S
    assert model.closed == 1
    assert redis.lists["session:s1:memory"] == []


# ---------------------------------------------------------------------------
# Tools through the agent
# ---------------------------------------------------------------------------


async def test_navigation_sends_valid_commands_and_one_model_call() -> None:
    model = FakeModel([call_chunks("navigate_to_product", {"product_id": "prod_001"})])
    events = await run_agent(model, user="Dẫn tôi tới chiếc áo xanh")

    (effect,) = effects(events)
    assert command_names(effect) == [
        "PLAY_AVATAR_ANIMATION", "NAVIGATE_TO_WAYPOINT", "HIGHLIGHT_PRODUCTS",
    ]
    validate_commands(effect)
    navigate = effect.commands[1]
    assert navigate.payload.waypoint_id == "WP_AISLE_01_SHIRTS"
    assert effect.commands[2].payload.product_ids == ["prod_001"]
    assert spoken(events) == "Mình dẫn bạn tới Blue Oxford Shirt nhé."
    assert len(model.calls) == 1  # the template replaced the second model call
    result = done(events)
    assert result.avatar_animation == "GESTURE_GUIDE"
    assert result.recommended_products == ["prod_001"]


async def test_navigating_to_a_category_walks_to_its_aisle() -> None:
    model = FakeModel([call_chunks("navigate_to_product", {"category": "áo khoác"})])
    events = await run_agent(model, user="Dẫn tôi đến khu áo khoác")

    (effect,) = effects(events)
    assert command_names(effect) == ["PLAY_AVATAR_ANIMATION", "NAVIGATE_TO_WAYPOINT"]
    validate_commands(effect)
    assert effect.commands[1].payload.waypoint_id == "WP_AISLE_02_JACKETS"
    assert "áo khoác" in spoken(events)


async def test_unknown_category_lists_what_the_store_has() -> None:
    model = FakeModel([call_chunks("navigate_to_product", {"category": "hats"})])
    events = await run_agent(model, language="en", user="take me to hats")
    assert effects(events) == []
    reply = spoken(events)
    assert "do not have that section" in reply and "Shoes" in reply


async def test_this_one_resolves_to_the_product_under_the_crosshair() -> None:
    model = FakeModel([call_chunks("open_product_inspector", {})])
    redis = FakeRedis(spatial={"gaze_product_id": "prod_002", "current_aisle": "A1"})
    events = await run_agent(model, user="Cho tôi xem 3D cái này", redis=redis)

    (effect,) = effects(events)
    assert effect.ui_action.action == "OPEN_PRODUCT_INSPECTOR"
    assert effect.ui_action.product_id == "prod_002"
    TypeAdapter(c.UiActionPayload).validate_python(effect.ui_action.model_dump())
    system_prompt = model.calls[0][1][0]["content"]
    assert "White Linen Shirt" in system_prompt and "prod_002" in system_prompt
    assert "White Linen Shirt" in spoken(events)


async def test_placeholder_ids_from_the_model_fall_back_to_the_gaze() -> None:
    model = FakeModel([call_chunks("open_product_inspector", {"product_id": "this"})])
    redis = FakeRedis(spatial={"gaze_product_id": "prod_004"})
    events = await run_agent(model, redis=redis)
    assert effects(events)[0].ui_action.product_id == "prod_004"


async def test_this_one_falls_back_to_the_product_discussed_last() -> None:
    memory = [json.dumps({"user": "u", "assistant": "a", "products": ["prod_005"]})]
    model = FakeModel([call_chunks("open_product_inspector", {})])
    events = await run_agent(model, redis=FakeRedis(memory=memory))
    assert effects(events)[0].ui_action.product_id == "prod_005"


async def test_unknown_or_missing_product_gets_a_polite_answer() -> None:
    nothing = await run_agent(FakeModel([call_chunks("open_product_inspector", {})]))
    assert "sản phẩm nào" in spoken(nothing)
    missing = await run_agent(
        FakeModel([call_chunks("add_to_cart", {"product_id": "prod_999"})]),
        language="en",
    )
    assert "could not find" in spoken(missing)


async def test_adding_without_a_size_asks_instead_of_guessing() -> None:
    gateway = FakeGateway()
    model = FakeModel([call_chunks("add_to_cart", {"product_id": "prod_001"})])
    events = await run_agent(model, gateway=gateway)

    assert spoken(events) == "Bạn muốn chọn Blue Oxford Shirt size S, M hoặc XL ạ?"
    assert effects(events) == []
    assert gateway.cart_lines == {}  # nothing was added


async def test_adding_without_a_size_in_english() -> None:
    model = FakeModel([call_chunks("add_to_cart", {"product_id": "prod_002"})])
    events = await run_agent(model, language="en")
    assert spoken(events) == (
        "Which size would you like for White Linen Shirt: S, M, L or XL?"
    )


async def test_adding_with_a_size_updates_the_cart() -> None:
    gateway = FakeGateway()
    args = {"product_id": "prod_001", "size": "m"}
    model = FakeModel([call_chunks("add_to_cart", args)])
    events = await run_agent(model, gateway=gateway, user="thêm size M")

    (effect,) = effects(events)
    assert effect.cart.item_count == 1 and effect.cart.total_amount == 450000
    TypeAdapter(c.CartSummary).validate_python(effect.cart.model_dump())
    assert gateway.cart_lines == {"var_001_M": 1}
    assert spoken(events) == (
        "Đã thêm 1 Blue Oxford Shirt size M vào giỏ. Giỏ có 1 món, tổng 450.000 đồng."
    )
    assert done(events).avatar_animation == "NOD"


async def test_quantity_given_as_text_and_english_price_format() -> None:
    args = {"product_id": "prod_002", "size": "Size L", "quantity": "2"}
    events = await run_agent(
        FakeModel([call_chunks("add_to_cart", args)]), language="en"
    )
    assert spoken(events) == (
        "Added 2 x White Linen Shirt, size L, to your cart. 2 items, "
        "total 1,040,000 VND."
    )


async def test_a_sold_out_size_offers_the_sizes_that_are_left() -> None:
    gateway = FakeGateway()
    args = {"product_id": "prod_001", "size": "L"}
    model = FakeModel([call_chunks("add_to_cart", args)])
    events = await run_agent(model, gateway=gateway)
    assert spoken(events) == (
        "Blue Oxford Shirt size L đã hết. Còn size S, M hoặc XL, bạn chọn size nào ạ?"
    )
    assert gateway.cart_lines == {}


async def test_asking_for_more_than_the_stock_says_how_many_are_left() -> None:
    args = {"product_id": "prod_001", "size": "XL", "quantity": 5}
    events = await run_agent(FakeModel([call_chunks("add_to_cart", args)]))
    assert spoken(events) == "Blue Oxford Shirt size XL chỉ còn 3 cái thôi ạ."


async def test_a_fully_sold_out_product_suggests_close_alternatives() -> None:
    args = {"product_id": "prod_003", "size": "M"}
    events = await run_agent(FakeModel([call_chunks("add_to_cart", args)]))
    assert spoken(events) == (
        "Tiếc quá, Red Flannel Shirt đã hết hàng. "
        "Bạn xem thử Brick Red Cotton Shirt nhé?"
    )
    assert done(events).recommended_products == ["prod_004"]
    assert done(events).avatar_animation == "SHRUG"


async def test_camera_search_for_a_sold_out_item_offers_an_alternative() -> None:
    gateway = FakeGateway()
    redis = FakeRedis(latest_cv=vision_json("#B22222", "shirt", 0.92))
    model = FakeModel([call_chunks("search_inventory", {"use_camera": True})])
    events = await run_agent(
        model, gateway=gateway, redis=redis, user="tìm cái giống cái này", language="en"
    )

    assert spoken(events) == (
        "Sorry, Red Flannel Shirt is sold out. How about Brick Red Cotton Shirt?"
    )
    assert done(events).recommended_products == ["prod_004"]
    assert gateway.search_calls[0]["category_slug"] == "shirts"  # from the camera label
    assert gateway.search_calls[0]["embedding"] is not None
    assert len(model.calls) == 1


async def test_search_with_stock_lets_the_model_word_the_answer() -> None:
    gateway = FakeGateway()
    model = FakeModel(
        [
            call_chunks("search_inventory", {"query_text": "jacket"}),
            text_chunks("Bên mình có áo khoác da đen giá 2.490.000 đồng ạ."),
        ]
    )
    events = await run_agent(model, gateway=gateway, user="Có áo khoác không?")

    assert spoken(events) == "Bên mình có áo khoác da đen giá 2.490.000 đồng ạ."
    assert len(model.calls) == 2
    second = model.calls[1][1]
    assistant, tool = second[-2], second[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["name"] == "search_inventory"
    assert assistant["tool_calls"][0]["args"] == {"query_text": "jacket"}
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_1"
    data = json.loads(tool["content"])
    assert data["status"] == "OK"
    assert data["results"][0]["product_id"] == "prod_005"
    assert done(events).recommended_products == ["prod_005"]


async def test_out_of_domain_object_searches_across_categories_by_colour() -> None:
    gateway = FakeGateway()
    redis = FakeRedis(latest_cv=vision_json("#808080", "mug", 0.03))
    model = FakeModel(
        [
            call_chunks("search_inventory", {"use_camera": True}),
            text_chunks("Mình không bán cốc, nhưng có đồ cùng tông xám."),
        ]
    )
    await run_agent(model, gateway=gateway, redis=redis, user="tìm cái giống cái này")

    assert gateway.search_calls[0]["category_slug"] is None
    camera = json.loads(model.calls[1][1][-1]["content"])["camera"]
    assert camera["is_store_product"] is False and camera["label"] == "mug"
    assert "does NOT sell" in model.calls[0][1][0]["content"]


async def test_camera_search_without_an_image_asks_to_show_the_item() -> None:
    model = FakeModel([call_chunks("search_inventory", {"use_camera": True})])
    events = await run_agent(model)
    assert "camera" in spoken(events)


async def test_search_with_no_results_says_so() -> None:
    model = FakeModel([call_chunks("search_inventory", {"query_text": "zzz"})])
    events = await run_agent(model, language="en")
    assert spoken(events) == "I could not find a matching product."


async def test_checkout_is_refused_for_an_empty_cart_and_opened_otherwise() -> None:
    gateway = FakeGateway()
    checkout = FakeModel([call_chunks("start_checkout", {})])
    empty = await run_agent(checkout, gateway=gateway)
    assert effects(empty) == []
    assert "trống" in spoken(empty)

    await gateway.add_to_cart("var_001_M", 2)
    checkout = FakeModel([call_chunks("start_checkout", {})])
    full = await run_agent(checkout, gateway=gateway)
    (effect,) = effects(full)
    assert effect.ui_action.action == "OPEN_CHECKOUT"
    assert spoken(full) == (
        "Mình mở trang thanh toán nhé. Giỏ có 2 món, tổng 900.000 đồng."
    )


async def test_product_details_go_back_to_the_model() -> None:
    model = FakeModel(
        [
            call_chunks("get_product_details", {"product_id": "prod_001"}),
            text_chunks("Áo này có size S, M và XL."),
        ]
    )
    events = await run_agent(model)
    data = json.loads(model.calls[1][1][-1]["content"])
    assert [s["size"] for s in data["sizes"] if s["in_stock"]] == ["S", "M", "XL"]
    assert spoken(events) == "Áo này có size S, M và XL."


async def test_two_tools_in_one_turn_are_both_run_and_both_spoken() -> None:
    gateway = FakeGateway()
    model = FakeModel(
        [
            call_chunks(
                "add_to_cart", {"product_id": "prod_002", "size": "M"}, "call_a", 0
            )
            + call_chunks(
                "navigate_to_product", {"product_id": "prod_002"}, "call_b", 1
            )
        ]
    )
    events = await run_agent(model, gateway=gateway, language="en")

    (effect,) = effects(events)
    assert effect.cart is not None
    assert command_names(effect)[1] == "NAVIGATE_TO_WAYPOINT"
    reply = spoken(events)
    assert "Added 1 x White Linen Shirt" in reply
    assert len(reply.split()) <= 40


async def test_text_before_a_tool_call_is_spoken_first() -> None:
    model = FakeModel(
        [
            text_chunks("Để mình xem nhé.")
            + call_chunks("navigate_to_product", {"product_id": "prod_001"})
        ]
    )
    events = await run_agent(model)
    assert isinstance(events[0], AgentTextDelta)
    assert spoken(events).startswith("Để mình xem nhé.")
    assert "Mình dẫn bạn tới Blue Oxford Shirt nhé." in spoken(events)


# ---------------------------------------------------------------------------
# Robustness against bad model output
# ---------------------------------------------------------------------------


async def test_arguments_split_across_many_chunks_are_reassembled() -> None:
    args = json.dumps({"product_id": "prod_001", "size": "M", "quantity": 2})
    fragments = [args[i : i + 3] for i in range(0, len(args), 3)]
    head = {"name": "add_to_cart", "args": "", "id": "c1", "index": 0}
    chunks = [Chunk(tool_call_chunks=[head])]
    chunks += [
        Chunk(tool_call_chunks=[{"name": None, "args": f, "id": None, "index": 0}])
        for f in fragments
    ]
    gateway = FakeGateway()
    await run_agent(FakeModel([chunks]), gateway=gateway)
    assert gateway.cart_lines == {"var_001_M": 2}


async def test_whole_tool_calls_without_fragments_also_work() -> None:
    chunk = Chunk(tool_calls=[{"name": "start_checkout", "args": {}, "id": "x1"}])
    events = await run_agent(FakeModel([[chunk]]))
    assert "trống" in spoken(events)


async def test_invalid_json_arguments_do_not_crash_the_turn() -> None:
    model = FakeModel(
        [call_chunks("add_to_cart", '{"product_id": "prod_001", "size":'),
         text_chunks("Xin lỗi, bạn nói lại giúp mình nhé.")]
    )
    gateway = FakeGateway()
    events = await run_agent(model, gateway=gateway)
    assert gateway.cart_lines == {}
    assert json.loads(model.calls[1][1][-1]["content"])["status"] == "ERROR"
    assert spoken(events) == "Xin lỗi, bạn nói lại giúp mình nhé."


async def test_unknown_tool_names_are_reported_back_to_the_model() -> None:
    model = FakeModel(
        [call_chunks("fly_to_moon", {}), text_chunks("Mình không làm được việc đó.")]
    )
    events = await run_agent(model)
    assert "unknown tool" in json.loads(model.calls[1][1][-1]["content"])["error"]
    assert spoken(events) == "Mình không làm được việc đó."


async def test_a_failing_tool_gives_an_apology_not_a_crash() -> None:
    model = FakeModel([call_chunks("get_product_details", {"product_id": "prod_001"})])
    events = await run_agent(model, gateway=FakeGateway(fail_on_get=True))
    assert "trục trặc" in spoken(events)
    assert isinstance(events[-1], AgentDone)


async def test_a_model_that_keeps_calling_tools_is_stopped() -> None:

    def search() -> list[Chunk]:
        return call_chunks("search_inventory", {"query_text": "shirt"})

    model = FakeModel([search(), search(), text_chunks("Đây là các mẫu áo sơ mi ạ.")])
    events = await run_agent(model, max_tool_rounds=2)

    assert [with_tools for with_tools, _ in model.calls] == [True, True, False]
    assert spoken(events) == "Đây là các mẫu áo sơ mi ạ."


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def test_earlier_turns_are_sent_to_the_model() -> None:
    memory = [
        json.dumps({"user": "Chào", "assistant": "Chào bạn", "products": []}),
        json.dumps(
            {"user": "Có áo không?", "assistant": "Có ạ", "products": ["prod_001"]}
        ),
    ]
    model = FakeModel([text_chunks("Vâng ạ.")])
    await run_agent(model, user="Giá bao nhiêu?", redis=FakeRedis(memory=memory))

    roles = [(m["role"], m["content"]) for m in model.calls[0][1][1:]]
    assert roles == [
        ("user", "Chào"), ("assistant", "Chào bạn"),
        ("user", "Có áo không?"), ("assistant", "Có ạ"),
        ("user", "Giá bao nhiêu?"),
    ]


async def test_memory_keeps_only_the_last_ten_turns() -> None:
    redis = FakeRedis()
    for turn in range(13):
        model = FakeModel([text_chunks(f"trả lời {turn}")])
        await run_agent(model, user=f"câu {turn}", redis=redis)
    stored = [json.loads(item) for item in redis.lists["session:s1:memory"]]
    assert len(stored) == agent_module.MEMORY_TURNS
    assert stored[0]["user"] == "câu 3" and stored[-1]["user"] == "câu 12"


async def test_memory_records_the_products_discussed() -> None:
    redis = FakeRedis()
    await run_agent(
        FakeModel([call_chunks("navigate_to_product", {"product_id": "prod_001"})]),
        redis=redis,
    )
    entry = json.loads(redis.lists["session:s1:memory"][0])
    assert entry["products"] == ["prod_001"]
    assert entry["assistant"] == "Mình dẫn bạn tới Blue Oxford Shirt nhé."


# ---------------------------------------------------------------------------
# Database-backed checks of the real gateway
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def seeded(db_engine: AsyncEngine) -> None:
    await seed(db_engine)


@pytest_asyncio.fixture
async def sql_gateway(db_session: AsyncSession, seeded: None) -> SqlStoreGateway:
    user = User(is_guest=True)
    db_session.add(user)
    await db_session.commit()
    return SqlStoreGateway(db_session, user.id)


async def test_sql_gateway_reads_products_and_categories(
    sql_gateway: SqlStoreGateway,
) -> None:
    product = await sql_gateway.get_product("prod_001")
    assert product is not None
    assert (product.name, product.base_price, product.waypoint_id) == (
        "Blue Oxford Shirt", 450000, "WP_AISLE_01_SHIRTS",
    )
    stock = [(v.size, v.stock) for v in product.variants]
    assert stock == [("S", 8), ("M", 5), ("L", 0), ("XL", 3)]
    assert await sql_gateway.get_product("prod_999") is None

    categories = {cat.slug: cat for cat in await sql_gateway.list_categories()}
    assert categories["jackets"].waypoint_id == "WP_AISLE_02_JACKETS"
    assert set(categories) == {"shirts", "jackets", "pants", "shoes"}


async def test_sql_gateway_search_and_alternatives(
    sql_gateway: SqlStoreGateway, db_session: AsyncSession
) -> None:
    from sqlalchemy import update

    from app.models.domain import Product

    unit = [1.0] + [0.0] * 511
    await db_session.execute(update(Product).values(image_embedding=unit))
    await db_session.commit()
    navy = lab_from_hex("#1F3A5F")
    results = await sql_gateway.search(
        query_text=None, lab=navy, embedding=unit, category_slug="shirts", limit=3
    )
    assert [r.product.product_id for r in results][0] == "prod_001"
    assert len(results) == 3

    red = lab_from_hex("#B22222")
    alternatives = await sql_gateway.alternatives_for("prod_003", red)
    assert [a.product.product_id for a in alternatives] == ["prod_004"]


async def test_sql_gateway_cart_round_trip(sql_gateway: SqlStoreGateway) -> None:
    cart = await sql_gateway.add_to_cart("var_001_M", 2)
    assert (cart.item_count, cart.total_amount) == (2, 900000)
    assert (await sql_gateway.cart_summary()).item_count == 2
    with pytest.raises(CartError) as caught:
        await sql_gateway.add_to_cart("var_001_M", 10)
    assert caught.value.code == "STOCK_INSUFFICIENT"


async def test_agent_end_to_end_on_the_real_database(
    sql_gateway: SqlStoreGateway, redis_client: Any
) -> None:
    model = FakeModel([call_chunks("navigate_to_product", {"product_id": "prod_001"})])
    agent = ShopkeeperAgent(model)
    events = [
        e async for e in agent.respond(
            gateway=sql_gateway, redis=redis_client, session_id="e2e",
            user_text="dẫn tôi tới áo sơ mi xanh", language="vi",
        )
    ]
    (effect,) = effects(events)
    assert effect.commands[1].payload.waypoint_id == "WP_AISLE_01_SHIRTS"
    stored = await redis_client.lrange("session:e2e:memory", 0, -1)
    assert json.loads(stored[0])["products"] == ["prod_001"]

    model = FakeModel([call_chunks("add_to_cart", {"product_id": "prod_001"})])
    events = [
        e async for e in ShopkeeperAgent(model).respond(
            gateway=sql_gateway, redis=redis_client, session_id="e2e",
            user_text="cho vào giỏ", language="vi",
        )
    ]
    assert spoken(events) == "Bạn muốn chọn Blue Oxford Shirt size S, M hoặc XL ạ?"


# ---------------------------------------------------------------------------
# Opt-in evaluation against a real model (needs an API key and network)
# ---------------------------------------------------------------------------

EVAL_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "walk to a section (vi)",
        "text": "Dẫn tôi đến khu áo khoác",
        "language": "vi",
        "tools_any": ["navigate_to_product"],
    },
    {
        "name": "walk to a section (en)",
        "text": "Take me to the jackets",
        "language": "en",
        "tools_any": ["navigate_to_product"],
    },
    {
        "name": "3D view of what I look at",
        "text": "Cho tôi xem 3D cái này",
        "language": "vi",
        "spatial": {"gaze_product_id": "prod_002"},
        "tools_any": ["open_product_inspector"],
    },
    {
        "name": "add 'this' with no size",
        "text": "Cho cái này vào giỏ",
        "language": "vi",
        "spatial": {"gaze_product_id": "prod_001"},
        "tools_any": ["add_to_cart"],
        "no_size": True,
    },
    {
        "name": "checkout",
        "text": "Tôi muốn thanh toán",
        "language": "vi",
        "tools_any": ["start_checkout"],
    },
    {
        "name": "mug in front of the camera",
        "text": "Tìm cái giống cái này",
        "language": "vi",
        "vision": vision_json("#808080", "mug", 0.03),
        "tools_any": ["search_inventory", None],
    },
    {
        "name": "shirt in front of the camera",
        "text": "Find me a shirt in this colour",
        "language": "en",
        "vision": vision_json("#B22222", "shirt", 0.92),
        "tools_any": ["search_inventory"],
    },
]



@pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1",
    reason="calls a real model; run with RUN_LLM_TESTS=1 and the provider's API key",
)
@pytest.mark.parametrize("scenario", EVAL_SCENARIOS, ids=lambda s: s["name"])
async def test_real_model_scenarios(scenario: dict[str, Any]) -> None:
    called: list[tuple[str, dict[str, Any]]] = []

    async def recording_runner(
        name: str, args: dict[str, Any], ctx: ToolContext, gateway: Any
    ) -> Any:
        called.append((name, args))
        return await run_tool(name, args, ctx, gateway)

    model = agent_module.create_chat_model()
    agent = ShopkeeperAgent(model, tool_runner=recording_runner)
    redis = FakeRedis(spatial=scenario.get("spatial"), latest_cv=scenario.get("vision"))
    started = time.perf_counter()
    first_event_at: float | None = None
    events = []
    async for event in agent.respond(
        gateway=FakeGateway(), redis=redis, session_id="s1",
        user_text=scenario["text"], language=scenario["language"],
    ):
        if first_event_at is None and not isinstance(event, AgentDone):
            first_event_at = time.perf_counter() - started
        events.append(event)
    total = time.perf_counter() - started

    reply = spoken(events)
    print(
        f"\n[{scenario['name']}] first output "
        f"{1000 * (first_event_at or total):.0f} ms, "
        f"total {1000 * total:.0f} ms, tools={[n for n, _ in called]}\n  -> {reply}"
    )
    assert reply, "the agent said nothing"
    assert len(reply.split()) <= 40
    names = [n for n, _ in called] or [None]
    assert any(n in scenario["tools_any"] for n in names), f"called {names}"
    if scenario.get("no_size"):
        assert all(not args.get("size") for n, args in called if n == "add_to_cart")
