"""The conversational shopkeeper: a streaming, tool-calling LangChain agent.

One turn (`ShopkeeperAgent.respond`):
  1. Load the session context from Redis: where the shopper is and what they are looking
     at (`spatial`), what the webcam last saw (`latest_cv`) and the last 10 turns
     (`memory`).
  2. Stream the model's answer. Text is forwarded as it arrives, capped at 40 words.
  3. If the model calls tools, run them. When every tool gives a ready-made reply, speak
     that and stop (no second model call); otherwise feed the tool results back to the
     model for the final wording.
  4. Store the turn in the sliding-window memory.

The agent yields events; the WebSocket layer turns them into TTS chunks, UNITY_COMMAND,
UI_ACTION and CART_UPDATED messages. Cancelling the consuming task (barge-in) stops the
model stream immediately and stores nothing.

Environment variables for the model (see `create_chat_model`):
    LLM_MODEL     "provider:model" for LangChain `init_chat_model`
                  (default "anthropic:claude-haiku-5-5"; try
                  "google_genai:gemini-3.5-flash-lite")
    LLM_API_KEY   optional; copied to the provider's standard variable if that is unset
    LLM_KWARGS    optional JSON object merged over the defaults, e.g.
                  '{"temperature": 0.2}'
"""

import asyncio
import contextlib
import functools
import json
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from app.services.agent_tools import (
    TOOL_SCHEMAS,
    SpatialContext,
    StoreGateway,
    ToolContext,
    ToolOutcome,
    VisionContext,
    run_tool,
    text,
)
from app.services.speech_service import Language

logger = logging.getLogger("app.agent")

MAX_WORDS = 40  # hard limit per turn; the prompt asks for fewer
PROMPT_WORDS = 35
MEMORY_TURNS = 10
MEMORY_TTL_SECONDS = 3600
MAX_TOOL_ROUNDS = 2
DEFAULT_LLM_MODEL = "anthropic:claude-haiku-5-5"

_TERMINAL_PUNCTUATION = ".!?…"


def spatial_key(session_id: str) -> str:
    return f"session:{session_id}:spatial"


def latest_cv_key(session_id: str) -> str:
    return f"session:{session_id}:latest_cv"


def memory_key(session_id: str) -> str:
    return f"session:{session_id}:memory"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentTextDelta:
    """A piece of the spoken reply, ready for TTS."""

    text: str


@dataclass(frozen=True)
class AgentEffects:
    """Client-side effects produced by tools (contract models)."""

    commands: list[Any] = field(default_factory=list)  # UnityCommand models
    ui_action: Any | None = None  # UiActionPayload model
    cart: Any | None = None  # CartSummary model


@dataclass(frozen=True)
class AgentDone:
    text: str
    recommended_products: list[str]
    avatar_animation: str | None


AgentEvent = AgentTextDelta | AgentEffects | AgentDone


# ---------------------------------------------------------------------------
# Word budget
# ---------------------------------------------------------------------------


class WordBudget:
    """Passes streamed text through until `max_words` words have been emitted.

    Words are runs of non-space characters (the same as `len(text.split())`), which
    also fits Vietnamese, where every syllable is a word. Whitespace is held back until
    the next word arrives, so the output never ends in a dangling space and a closing
    full stop attaches directly to the last word.
    """

    def __init__(self, max_words: int) -> None:
        self._max = max_words
        self._count = 0
        self._in_word = False
        self._pending = ""  # whitespace waiting for the next word
        self._last = ""
        self.exhausted = False

    def feed(self, delta: str) -> str:
        if self.exhausted:
            return ""
        kept: list[str] = []
        for char in delta:
            if char.isspace():
                self._in_word = False
                self._pending = self._pending or char
                continue
            if not self._in_word:
                if self._count >= self._max:
                    self.exhausted = True
                    self._pending = ""
                    break
                self._count += 1
                self._in_word = True
            if self._pending:
                kept.append(self._pending)
                self._pending = ""
            kept.append(char)
        emitted = "".join(kept)
        if emitted:
            self._last = emitted[-1]
        return emitted

    def finish(self) -> str:
        """A full stop to add when the text was cut off in the middle of a sentence."""
        if self.exhausted and self._last and self._last not in _TERMINAL_PUNCTUATION:
            return "."
        return ""


# ---------------------------------------------------------------------------
# Session context
# ---------------------------------------------------------------------------


def parse_spatial(raw: dict[str, str] | None) -> SpatialContext:
    raw = raw or {}

    def number(key: str) -> float | None:
        try:
            return float(raw[key])
        except (KeyError, TypeError, ValueError):
            return None

    x, y, z = number("pos_x"), number("pos_y"), number("pos_z")
    position = (x, y, z) if x is not None and y is not None and z is not None else None
    return SpatialContext(
        current_aisle=raw.get("current_aisle") or None,
        gaze_product_id=raw.get("gaze_product_id") or None,
        position=position,
    )


def parse_vision(raw: str | None) -> VisionContext | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        lab = data["lab"]
        embedding = data.get("embedding")
        return VisionContext(
            lab=(float(lab[0]), float(lab[1]), float(lab[2])),
            color_name=str(data["color_name"]),
            category_label=str(data["category_label"]),
            category_confidence=float(data["category_confidence"]),
            quality_warning=data.get("quality_warning"),
            embedding=tuple(float(v) for v in embedding) if embedding else None,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("Ignoring malformed latest_cv value")
        return None


def parse_history(items: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    """Memory entries -> chat messages, plus the product discussed most recently."""
    messages: list[dict[str, Any]] = []
    last_product: str | None = None
    for item in items:
        try:
            entry = json.loads(item)
            user, assistant = str(entry["user"]), str(entry["assistant"])
        except (KeyError, TypeError, ValueError):
            continue
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
        products = entry.get("products")
        if isinstance(products, list) and products:
            last_product = str(products[0])
    return messages, last_product


_LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English"}


def build_system_prompt(
    ctx: ToolContext, gaze_name: str | None, last_name: str | None
) -> str:
    lines = [
        "You are the friendly shopkeeper of a 3D clothing store and you talk with "
        "shoppers by voice.",
        f"Reply in {_LANGUAGE_NAMES[ctx.language]}, in at most {PROMPT_WORDS} words "
        "(one or two short sentences). Plain speech only: no markdown, lists, emoji "
        "or stage directions. Ask at most one question.",
        "Rules:",
        "- Never invent products, prices, sizes or stock. Use the tools for facts.",
        "- For walking the shopper somewhere, adding to the cart, opening the 3D view "
        "or checkout: call the tool at once and write no text before it. The system "
        "speaks the confirmation itself.",
        "- To walk to a whole section (for example jackets) call navigate_to_product "
        "with category. To find a product first, call search_inventory.",
        "- If the shopper does not say a size, call add_to_cart without a size; the "
        "tool asks which size they want. Never choose a size yourself.",
        "- 'This one', 'that one' or 'the one on the left' mean the product named in "
        "the context below; pass its product_id, or omit product_id.",
        "- After a tool returns data, answer from that data only. Mention honestly if "
        "something is sold out.",
        "Context:",
    ]

    spatial = ctx.spatial
    if spatial.current_aisle:
        lines.append(f"- The shopper is in aisle {spatial.current_aisle}.")
    if spatial.gaze_product_id:
        label = f"{gaze_name} " if gaze_name else ""
        lines.append(
            f"- The shopper is looking at: {label}(product_id "
            f"{spatial.gaze_product_id})."
        )
    elif ctx.last_product_id:
        label = f"{last_name} " if last_name else ""
        lines.append(
            f"- The product discussed last: {label}(product_id {ctx.last_product_id})."
        )
    else:
        lines.append("- No product is being looked at or discussed yet.")

    vision = ctx.vision
    if vision is None:
        lines.append("- Camera: nothing is being shown.")
    elif vision.is_in_domain:
        lines.append(
            f"- Camera: the shopper is showing a {vision.category_label} (it looks "
            f"like something the store sells), main colour {vision.color_name}. To "
            "find matches call search_inventory with use_camera=true."
        )
    else:
        lines.append(
            f"- Camera: the shopper is showing something the store does NOT sell (it "
            f"looks like a {vision.category_label}), main colour {vision.color_name}. "
            "If they ask about it, politely say the store does not sell that and offer "
            "store items in a similar colour by calling search_inventory with "
            "use_camera=true."
        )
    if vision is not None and vision.quality_warning == "LOW_LIGHT":
        lines.append("- The camera image is too dark: ask for better light.")
    elif vision is not None and vision.quality_warning == "OVEREXPOSED":
        lines.append("- The camera image is too bright: ask them to reduce the glare.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model output parsing
# ---------------------------------------------------------------------------


def chunk_text(chunk: Any) -> str:
    """Text of a streamed chunk (a string, or a list of content blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _merge_tool_chunks(
    calls: list[dict[str, Any]], by_index: dict[int, int], chunks: list[Any]
) -> None:
    """Assemble streamed tool-call fragments (name/id first, JSON args in pieces)."""
    for piece in chunks:
        index = piece.get("index")
        name = piece.get("name")
        if name and (index is None or index not in by_index):
            calls.append(
                {"id": piece.get("id"), "name": name, "args": piece.get("args") or ""}
            )
            if index is not None:
                by_index[index] = len(calls) - 1
            continue
        position = by_index.get(index) if index is not None else None
        if position is not None:
            target: dict[str, Any] | None = calls[position]
        else:
            target = calls[-1] if calls else None
        if target is None:
            continue
        if piece.get("id") and not target["id"]:
            target["id"] = piece["id"]
        target["args"] += piece.get("args") or ""


def _finish_call(call: dict[str, Any], number: int) -> dict[str, Any]:
    try:
        parsed = json.loads(call["args"] or "{}")
        invalid = not isinstance(parsed, dict)
    except ValueError:
        parsed, invalid = {}, True
    return {
        "id": call["id"] or f"call_{number}",
        "name": call["name"],
        "args": {} if invalid else parsed,
        "invalid": invalid,
    }


async def _aclose(stream: Any) -> None:
    closer = getattr(stream, "aclose", None)
    if closer is not None:
        with contextlib.suppress(Exception):
            await closer()


async def stream_turn(
    model: Any, messages: list[dict[str, Any]]
) -> AsyncGenerator[tuple[str, Any], None]:
    """Run one model call.

    Yields ("text", delta) pieces, then one ("calls", [tool calls]) item.
    """
    calls: list[dict[str, Any]] = []
    by_index: dict[int, int] = {}
    complete: list[dict[str, Any]] = []
    stream = model.astream(messages)
    try:
        async for chunk in stream:
            delta = chunk_text(chunk)
            if delta:
                yield ("text", delta)
            fragments = getattr(chunk, "tool_call_chunks", None) or []
            _merge_tool_chunks(calls, by_index, fragments)
            complete.extend(getattr(chunk, "tool_calls", None) or [])
    finally:
        await _aclose(stream)

    if not calls and complete:  # a model that returns whole tool calls, not fragments
        calls = [
            {
                "id": item.get("id"),
                "name": item["name"],
                "args": json.dumps(item["args"]),
            }
            for item in complete
            if item.get("name")
        ]
    yield ("calls", [_finish_call(call, i) for i, call in enumerate(calls)])


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

ToolRunner = Callable[
    [str, dict[str, Any], ToolContext, StoreGateway], Awaitable[ToolOutcome]
]


async def _await(value: Any) -> Any:
    return await cast("Awaitable[Any]", value)


class ShopkeeperAgent:
    def __init__(
        self,
        model: Any,
        tool_runner: ToolRunner = run_tool,
        max_words: int = MAX_WORDS,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._model = model
        self._tool_model = model.bind_tools(TOOL_SCHEMAS)
        self._run_tool = tool_runner
        self._max_words = max_words
        self._max_tool_rounds = max_tool_rounds

    async def _load_state(
        self, gateway: StoreGateway, redis: Any, session_id: str, language: Language
    ) -> tuple[ToolContext, list[dict[str, Any]], str | None, str | None]:
        spatial_raw, vision_raw, memory_raw = await asyncio.gather(
            _await(redis.hgetall(spatial_key(session_id))),
            _await(redis.get(latest_cv_key(session_id))),
            _await(redis.lrange(memory_key(session_id), -MEMORY_TURNS, -1)),
        )
        history, last_product = parse_history(list(memory_raw or []))
        ctx = ToolContext(
            language=language,
            spatial=parse_spatial(spatial_raw),
            vision=parse_vision(vision_raw),
            last_product_id=last_product,
        )

        async def name_of(product_id: str | None) -> str | None:
            if not product_id:
                return None
            product = await gateway.get_product(product_id)
            return product.name if product else None

        gaze_name = await name_of(ctx.spatial.gaze_product_id)
        last_name = await name_of(None if gaze_name else ctx.last_product_id)
        return ctx, history, gaze_name, last_name

    async def _execute(
        self, call: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
    ) -> ToolOutcome:
        if call["invalid"]:
            return ToolOutcome(
                data={"status": "ERROR", "error": "tool arguments were not valid JSON"}
            )
        return await self._run_tool(call["name"], call["args"], ctx, gateway)

    async def respond(
        self,
        *,
        gateway: StoreGateway,
        redis: Any,
        session_id: str,
        user_text: str,
        language: Language,
    ) -> AsyncGenerator[AgentEvent, None]:
        ctx, history, gaze_name, last_name = await self._load_state(
            gateway, redis, session_id, language
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(ctx, gaze_name, last_name),
            },
            *history,
            {"role": "user", "content": user_text},
        ]
        budget = WordBudget(self._max_words)
        spoken: list[str] = []
        products: list[str] = []
        animation: str | None = None

        def speak(delta: str) -> str:
            piece = budget.feed(delta)
            if piece:
                spoken.append(piece)
            return piece

        for round_number in range(self._max_tool_rounds + 1):
            can_use_tools = round_number < self._max_tool_rounds
            model = self._tool_model if can_use_tools else self._model
            raw_text: list[str] = []
            calls: list[dict[str, Any]] = []
            turn = stream_turn(model, messages)
            try:
                async for kind, payload in turn:
                    if kind == "text":
                        raw_text.append(payload)
                        piece = speak(payload)
                        if piece:
                            yield AgentTextDelta(piece)
                    else:
                        calls = payload
            finally:
                await turn.aclose()
            if not calls:
                break

            outcomes = [await self._execute(call, ctx, gateway) for call in calls]
            effects = AgentEffects(
                commands=[cmd for o in outcomes for cmd in o.commands],
                ui_action=next(
                    (o.ui_action for o in reversed(outcomes) if o.ui_action), None
                ),
                cart=next((o.cart for o in reversed(outcomes) if o.cart), None),
            )
            if effects.commands or effects.ui_action or effects.cart:
                yield effects
            for outcome in outcomes:
                products.extend(p for p in outcome.products if p not in products)
                animation = outcome.animation or animation

            if all(o.reply for o in outcomes):
                separator = " " if spoken else ""
                reply = separator + " ".join(o.reply for o in outcomes if o.reply)
                piece = speak(reply)
                if piece:
                    yield AgentTextDelta(piece)
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(raw_text),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "name": call["name"],
                            "args": call["args"],
                            "type": "tool_call",
                        }
                        for call in calls
                    ],
                }
            )
            for call, outcome in zip(calls, outcomes, strict=True):
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(outcome.data, ensure_ascii=False),
                        "tool_call_id": call["id"],
                        "name": call["name"],
                    }
                )

        closing = budget.finish()
        if closing:
            spoken.append(closing)
            yield AgentTextDelta(closing)
        if not "".join(spoken).strip():
            fallback = text(language, "fallback")
            spoken.append(fallback)
            yield AgentTextDelta(fallback)

        final_text = "".join(spoken).strip()
        await self._remember(redis, session_id, user_text, final_text, products)
        yield AgentDone(final_text, products, animation)

    @staticmethod
    async def _remember(
        redis: Any, session_id: str, user_text: str, reply: str, products: list[str]
    ) -> None:
        key = memory_key(session_id)
        entry = json.dumps(
            {"user": user_text, "assistant": reply, "products": products},
            ensure_ascii=False,
        )
        await _await(redis.rpush(key, entry))
        await _await(redis.ltrim(key, -MEMORY_TURNS, -1))
        await _await(redis.expire(key, MEMORY_TTL_SECONDS))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_API_KEY_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def default_model_kwargs(provider: str, model_name: str) -> dict[str, Any]:
    """Settings that keep time-to-first-token low (reasoning/thinking OFF).

    Fast-tier models from all three vendors take many seconds to start answering when
    their reasoning mode is on, so it is disabled here.
    """
    kwargs: dict[str, Any] = {"timeout": 10, "max_retries": 1}
    if provider == "anthropic":
        kwargs.update(
            temperature=0.3, max_tokens=220, thinking={"type": "disabled"}
        )
    elif provider == "google_genai":
        kwargs.update(temperature=0.3, max_output_tokens=220)
        if "lite" not in model_name:  # Flash-Lite already defaults to minimal
            kwargs["thinking_level"] = "minimal"
    return kwargs


def create_chat_model() -> Any:
    """Build the chat model named by LLM_MODEL ("provider:model")."""
    from langchain.chat_models import init_chat_model

    spec = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
    provider, _, model_name = spec.partition(":")
    api_key = os.environ.get("LLM_API_KEY")
    variable = _API_KEY_VARIABLES.get(provider)
    if api_key and variable:
        os.environ.setdefault(variable, api_key)

    kwargs = default_model_kwargs(provider, model_name)
    extra = os.environ.get("LLM_KWARGS")
    if extra:
        kwargs.update(json.loads(extra))
    return init_chat_model(spec, **kwargs)


@functools.cache
def get_shopkeeper_agent() -> ShopkeeperAgent:
    """Process-wide agent using the model from the environment."""
    return ShopkeeperAgent(create_chat_model())
