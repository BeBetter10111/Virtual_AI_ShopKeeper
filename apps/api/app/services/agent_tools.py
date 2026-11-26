"""Tools for the shopkeeper agent (Tool Contract in ARCHITECTURE.md section 4.4).

Each tool returns a `ToolOutcome`: data for the language model, side effects for the
client (Unity commands, UI action, cart update) and, for outcomes whose wording does
not need creativity, a ready-made spoken `reply`. When every tool in a round has a
reply, the agent speaks those and skips the second model call, which saves a full
model round trip and keeps stock, size and price statements exact.

Database access goes through the `StoreGateway` interface. `SqlStoreGateway` is the
real implementation; tests use an in-memory one.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import Category, Product, ShelfLocation
from app.schemas import contracts as c
from app.services import cart_service
from app.services.cart_service import CartError
from app.services.inventory_search import pick_alternatives, rank_products
from app.services.speech_service import Language
from app.services.vision_service import DEFAULT_LABELS, DOMAIN_CONFIDENCE_THRESHOLD

logger = logging.getLogger("app.agent")

SEARCH_RESULT_LIMIT = 3
MAX_QUANTITY = 20

Lab = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Plain data used by the tools (independent of the database layer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantInfo:
    variant_id: str
    size: str
    color_name: str
    stock: int
    price_delta: int


@dataclass(frozen=True)
class ProductInfo:
    product_id: str
    name: str
    description: str
    base_price: int
    category_slug: str
    category_name: str
    waypoint_id: str
    aisle_code: str
    color_lab: Lab | None
    variants: tuple[VariantInfo, ...]

    @property
    def in_stock_variants(self) -> list[VariantInfo]:
        return [v for v in self.variants if v.stock > 0]

    @property
    def in_stock(self) -> bool:
        return bool(self.in_stock_variants)

    @property
    def color_name(self) -> str:
        return self.variants[0].color_name if self.variants else ""


@dataclass(frozen=True)
class RankedProduct:
    product: ProductInfo
    delta_e00: float | None
    score: float


@dataclass(frozen=True)
class CategoryInfo:
    slug: str
    name: str
    waypoint_id: str | None
    aisle_code: str


class StoreGateway(Protocol):
    async def get_product(self, product_id: str) -> ProductInfo | None: ...

    async def list_categories(self) -> list[CategoryInfo]: ...

    async def search(
        self,
        *,
        query_text: str | None,
        lab: Sequence[float] | None,
        embedding: Sequence[float] | None,
        category_slug: str | None,
        limit: int,
    ) -> list[RankedProduct]: ...

    async def alternatives_for(
        self, product_id: str, lab: Sequence[float] | None
    ) -> list[RankedProduct]:
        """In-stock, same-category products close in colour to `lab` (or to the
        product itself when `lab` is None)."""
        ...

    async def add_to_cart(self, variant_id: str, quantity: int) -> c.CartSummary:
        """Raises `CartError` (e.g. STOCK_INSUFFICIENT)."""
        ...

    async def cart_summary(self) -> c.CartSummary: ...


class SqlStoreGateway:
    """StoreGateway backed by PostgreSQL, for one user."""

    def __init__(self, db: AsyncSession, user_id: Any) -> None:
        self._db = db
        self._user_id = user_id

    @staticmethod
    def _to_info(product: Product) -> ProductInfo:
        lab = product.color_lab
        variants = sorted(
            product.variants, key=lambda v: cart_service.size_sort_key(v.size)
        )
        return ProductInfo(
            product_id=product.public_id,
            name=product.name,
            description=product.description,
            base_price=int(product.base_price),
            category_slug=product.category.slug,
            category_name=product.category.name,
            waypoint_id=product.shelf_location.unity_waypoint_id,
            aisle_code=product.shelf_location.aisle_code,
            color_lab=(float(lab[0]), float(lab[1]), float(lab[2])) if lab else None,
            variants=tuple(
                VariantInfo(
                    variant_id=v.public_id,
                    size=v.size,
                    color_name=v.color_name,
                    stock=v.stock_qty,
                    price_delta=int(v.price_delta),
                )
                for v in variants
            ),
        )

    async def get_product(self, product_id: str) -> ProductInfo | None:
        row = (
            await self._db.execute(
                select(Product).where(Product.public_id == product_id)
            )
        ).scalar_one_or_none()
        return None if row is None else self._to_info(row)

    async def list_categories(self) -> list[CategoryInfo]:
        categories = (
            (await self._db.execute(select(Category).order_by(Category.slug)))
            .scalars()
            .all()
        )
        shelves = (await self._db.execute(select(ShelfLocation))).scalars().all()
        waypoints = {s.aisle_code: s.unity_waypoint_id for s in shelves}
        return [
            CategoryInfo(
                slug=cat.slug,
                name=cat.name,
                waypoint_id=waypoints.get(cat.aisle_zone_code),
                aisle_code=cat.aisle_zone_code,
            )
            for cat in categories
        ]

    async def search(
        self,
        *,
        query_text: str | None,
        lab: Sequence[float] | None,
        embedding: Sequence[float] | None,
        category_slug: str | None,
        limit: int,
    ) -> list[RankedProduct]:
        ranked = await rank_products(
            self._db,
            query_text=query_text,
            lab=lab,
            embedding=embedding,
            category_slug=category_slug,
        )
        return [
            RankedProduct(self._to_info(s.product), s.delta_e00, s.score)
            for s in ranked[:limit]
        ]

    async def alternatives_for(
        self, product_id: str, lab: Sequence[float] | None
    ) -> list[RankedProduct]:
        product = await self.get_product(product_id)
        if product is None:
            return []
        reference = lab if lab is not None else product.color_lab
        if reference is None:
            return []
        ranked = await rank_products(
            self._db, lab=reference, category_slug=product.category_slug
        )
        best = next((s for s in ranked if s.product.public_id == product_id), None)
        if best is None:
            return []
        return [
            RankedProduct(self._to_info(s.product), s.delta_e00, s.score)
            for s in pick_alternatives(ranked, best)
        ]

    async def add_to_cart(self, variant_id: str, quantity: int) -> c.CartSummary:
        return await cart_service.add_item(
            self._db, self._user_id, variant_id, quantity, None
        )

    async def cart_summary(self) -> c.CartSummary:
        return await cart_service.get_cart_summary(self._db, self._user_id)


# ---------------------------------------------------------------------------
# Context and outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialContext:
    current_aisle: str | None = None
    gaze_product_id: str | None = None
    position: Lab | None = None


@dataclass(frozen=True)
class VisionContext:
    """What the webcam last saw (Redis `latest_cv`, which expires after 30 s)."""

    lab: Lab
    color_name: str
    category_label: str
    category_confidence: float
    quality_warning: str | None = None
    embedding: tuple[float, ...] | None = None

    @property
    def is_in_domain(self) -> bool:
        return self.category_confidence >= DOMAIN_CONFIDENCE_THRESHOLD

    @property
    def category_slug(self) -> str | None:
        for spec in DEFAULT_LABELS:
            if spec.label == self.category_label:
                return spec.category_slug
        return None


@dataclass(frozen=True)
class ToolContext:
    language: Language
    spatial: SpatialContext = field(default_factory=SpatialContext)
    vision: VisionContext | None = None
    last_product_id: str | None = None

    @property
    def focus_product_id(self) -> str | None:
        """The product "this one" means: the one under the crosshair, else the last
        one discussed."""
        return self.spatial.gaze_product_id or self.last_product_id


@dataclass
class ToolOutcome:
    data: dict[str, Any]
    reply: str | None = None
    commands: list[Any] = field(default_factory=list)
    ui_action: Any | None = None
    cart: Any | None = None
    products: list[str] = field(default_factory=list)
    animation: str | None = None


# ---------------------------------------------------------------------------
# Localised wording
# ---------------------------------------------------------------------------

_CATEGORY_WORDS: dict[Language, dict[str, str]] = {
    "vi": {
        "shirts": "áo sơ mi",
        "jackets": "áo khoác",
        "pants": "quần",
        "shoes": "giày",
    },
    "en": {
        "shirts": "shirts",
        "jackets": "jackets",
        "pants": "pants",
        "shoes": "shoes",
    },
}

_CATEGORY_ALIASES = {
    "shirt": "shirts",
    "shirts": "shirts",
    "áo sơ mi": "shirts",
    "sơ mi": "shirts",
    "áo": "shirts",
    "jacket": "jackets",
    "jackets": "jackets",
    "coat": "jackets",
    "áo khoác": "jackets",
    "khoác": "jackets",
    "pants": "pants",
    "trousers": "pants",
    "jeans": "pants",
    "quần": "pants",
    "shoe": "shoes",
    "shoes": "shoes",
    "sneakers": "shoes",
    "boots": "shoes",
    "giày": "shoes",
}

_PLACEHOLDER_IDS = frozenset(
    {"", "this", "that", "it", "current", "gaze", "focused", "none", "null",
     "cái này", "cái kia"}
)

_TEXT: dict[Language, dict[str, str]] = {
    "vi": {
        "navigate_product": "Mình dẫn bạn tới {name} nhé.",
        "navigate_category": "Mình dẫn bạn tới khu {category} nhé.",
        "inspector": "Mình mở chế độ xem 3D của {name} cho bạn nhé.",
        "checkout": "Mình mở trang thanh toán nhé. Giỏ có {count} món, tổng {total}.",
        "checkout_empty": "Giỏ hàng của bạn đang trống. Bạn muốn xem món nào không ạ?",
        "needs_variant": "Bạn muốn chọn {name} size {options} ạ?",
        "added": "Đã thêm {quantity} {name} size {size} vào giỏ. Giỏ có {count} món, "
        "tổng {total}.",
        "size_sold_out": "{name} size {size} đã hết. Còn size {options}, bạn chọn size "
        "nào ạ?",
        "product_sold_out": "Tiếc quá, {name} đã hết hàng. Bạn xem thử {alternatives} "
        "nhé?",
        "product_sold_out_none": "Tiếc quá, {name} đã hết hàng và mình chưa có mẫu nào "
        "gần màu này.",
        "stock_limited": "{name} size {size} chỉ còn {available} cái thôi ạ.",
        "need_product": "Bạn đang nói về sản phẩm nào ạ?",
        "not_found": "Mình không tìm thấy sản phẩm đó trong cửa hàng.",
        "unknown_category": "Mình chưa có khu vực đó. Cửa hàng có {categories}.",
        "no_camera": "Bạn giơ món đồ lên trước camera giúp mình nhé.",
        "no_results": "Mình chưa tìm thấy sản phẩm nào phù hợp.",
        "tool_failed": "Xin lỗi, mình gặp chút trục trặc. Bạn thử lại giúp mình nhé.",
        "fallback": "Xin lỗi, mình chưa nghe rõ. Bạn nói lại giúp mình nhé?",
        "price": "{amount} đồng",
        "or": "hoặc",
    },
    "en": {
        "navigate_product": "Let me take you to {name}.",
        "navigate_category": "Let me take you to the {category} section.",
        "inspector": "Opening the 3D view of {name} for you.",
        "checkout": "Opening checkout. Your cart has {count} items, total {total}.",
        "checkout_empty": "Your cart is empty. Would you like to look at something?",
        "needs_variant": "Which size would you like for {name}: {options}?",
        "added": "Added {quantity} x {name}, size {size}, to your cart. {count} items, "
        "total {total}.",
        "size_sold_out": "{name} in size {size} is sold out. Available sizes: "
        "{options}. Which one would you like?",
        "product_sold_out": "Sorry, {name} is sold out. How about {alternatives}?",
        "product_sold_out_none": "Sorry, {name} is sold out and I have nothing similar "
        "in that colour.",
        "stock_limited": "Only {available} left of {name} in size {size}.",
        "need_product": "Which product do you mean?",
        "not_found": "I could not find that product in the store.",
        "unknown_category": "We do not have that section. We have {categories}.",
        "no_camera": "Please hold the item up to the camera.",
        "no_results": "I could not find a matching product.",
        "tool_failed": "Sorry, something went wrong. Please try again.",
        "fallback": "Sorry, I did not catch that. Could you say it again?",
        "price": "{amount} VND",
        "or": "or",
    },
}


def text(language: Language, key: str, **values: Any) -> str:
    return _TEXT[language][key].format(**values)


def format_price(amount: int, language: Language) -> str:
    if language == "vi":
        return text(language, "price", amount=f"{amount:,}".replace(",", "."))
    return text(language, "price", amount=f"{amount:,}")


def join_options(items: Sequence[str], language: Language) -> str:
    """"S, M or L" / "S, M hoặc L"."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {_TEXT[language]['or']} {items[-1]}"


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function format, accepted by every LangChain chat model)
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties},
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "search_inventory",
        "Find products in the store. Use it for any question about what the store "
        "sells, colours, or what matches the item the shopper holds up to the camera.",
        {
            "query_text": {
                "type": "string",
                "description": "Words to look for in product names and descriptions.",
            },
            "category": {
                "type": "string",
                "description": "Limit to one category: shirts, jackets, pants or "
                "shoes.",
            },
            "color_lab": {
                "type": "array",
                "items": {"type": "number"},
                "description": "CIELAB colour [L, a, b] to match. Prefer use_camera.",
            },
            "use_camera": {
                "type": "boolean",
                "description": "Match the colour and look of the item the shopper is "
                "currently showing to the camera.",
            },
        },
    ),
    _tool(
        "get_product_details",
        "Get the price, sizes and stock of one product.",
        {"product_id": {"type": "string", "description": "e.g. prod_001"}},
    ),
    _tool(
        "navigate_to_product",
        "Walk the shopper to a product's shelf, or to a whole category section. "
        "Call it immediately, without writing any text first.",
        {
            "product_id": {"type": "string", "description": "e.g. prod_001"},
            "category": {
                "type": "string",
                "description": "Section to walk to: shirts, jackets, pants or shoes. "
                "Use this when the shopper names a section, not a product.",
            },
        },
    ),
    _tool(
        "add_to_cart",
        "Add a product to the cart. If the shopper did not say a size, call it "
        "without a size: the tool asks them. Call it immediately, without writing "
        "any text first.",
        {
            "product_id": {
                "type": "string",
                "description": "Omit when the shopper says 'this one' (uses the "
                "product they are looking at).",
            },
            "variant_id": {"type": "string", "description": "e.g. var_001_M"},
            "size": {"type": "string", "description": "e.g. S, M, L, XL, 41"},
            "quantity": {"type": "integer", "description": "Default 1."},
        },
    ),
    _tool(
        "open_product_inspector",
        "Open the 3D viewer for a product. Call it immediately, without any text.",
        {
            "product_id": {
                "type": "string",
                "description": "Omit for 'this one' (the product being looked at).",
            }
        },
    ),
    _tool(
        "start_checkout",
        "Open the checkout screen for the current cart. Call it immediately, "
        "without any text.",
        {},
    ),
]


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def _arg_str(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _arg_int(
    args: dict[str, Any], key: str, default: int, minimum: int, maximum: int
) -> int:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return max(minimum, min(maximum, int(float(value))))
    except ValueError:
        return default


def _arg_bool(args: dict[str, Any], key: str) -> bool:
    value = args.get(key)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _arg_lab(args: dict[str, Any]) -> Lab | None:
    value = args.get("color_lab")
    if isinstance(value, list) and len(value) == 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
    return None


async def _resolve_category(raw: str | None, gateway: StoreGateway) -> str | None:
    """Map a free-text category (Vietnamese or English) to a store category slug."""
    if raw is None:
        return None
    key = raw.strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    for category in await gateway.list_categories():
        if key in {category.slug.lower(), category.name.lower()}:
            return category.slug
    return None


async def _resolve_product(
    raw: str | None, ctx: ToolContext, gateway: StoreGateway
) -> tuple[ProductInfo | None, str | None]:
    """Find the product a tool call is about; returns (product, error_key)."""
    product_id = raw.strip() if raw else None
    if product_id and product_id.lower() in _PLACEHOLDER_IDS:
        product_id = None
    product_id = product_id or ctx.focus_product_id
    if not product_id:
        return None, "need_product"
    product = await gateway.get_product(product_id)
    return (product, None) if product else (None, "not_found")


def _error_outcome(key: str, ctx: ToolContext, **values: Any) -> ToolOutcome:
    return ToolOutcome(
        data={"status": "ERROR", "error": key},
        reply=text(ctx.language, key, **values),
    )


def _product_summary(item: RankedProduct) -> dict[str, Any]:
    product = item.product
    summary: dict[str, Any] = {
        "product_id": product.product_id,
        "name": product.name,
        "price": product.base_price,
        "category": product.category_slug,
        "color": product.color_name,
        "in_stock": product.in_stock,
        "sizes_in_stock": [v.size for v in product.in_stock_variants],
        "waypoint_id": product.waypoint_id,
    }
    if item.delta_e00 is not None:
        summary["color_distance_de00"] = round(item.delta_e00, 1)
    return summary


def _sold_out_outcome(
    product: ProductInfo, alternatives: list[RankedProduct], ctx: ToolContext
) -> ToolOutcome:
    """The product is sold out in every size: say so and suggest close alternatives."""
    language = ctx.language
    if alternatives:
        names = join_options([a.product.name for a in alternatives], language)
        reply = text(
            language, "product_sold_out", name=product.name, alternatives=names
        )
    else:
        reply = text(language, "product_sold_out_none", name=product.name)
    return ToolOutcome(
        data={
            "status": "SOLD_OUT",
            "product_id": product.product_id,
            "alternatives": [_product_summary(a) for a in alternatives],
        },
        reply=reply,
        products=[a.product.product_id for a in alternatives],
        animation="SHRUG",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def search_inventory(
    args: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
) -> ToolOutcome:
    query_text = _arg_str(args, "query_text")
    category_raw = _arg_str(args, "category")
    category = await _resolve_category(category_raw, gateway)
    if category_raw is not None and category is None:
        categories = await gateway.list_categories()
        names = join_options([cat.name for cat in categories], ctx.language)
        return _error_outcome("unknown_category", ctx, categories=names)

    lab = _arg_lab(args)
    embedding: Sequence[float] | None = None
    camera: dict[str, Any] | None = None
    if _arg_bool(args, "use_camera"):
        vision = ctx.vision
        if vision is None:
            return _error_outcome("no_camera", ctx)
        lab, embedding = vision.lab, vision.embedding
        camera = {
            "label": vision.category_label,
            "is_store_product": vision.is_in_domain,
            "color_name": vision.color_name,
            "quality_warning": vision.quality_warning,
        }
        if category is None and vision.is_in_domain:
            category = vision.category_slug

    results = await gateway.search(
        query_text=query_text,
        lab=lab,
        embedding=embedding,
        category_slug=category,
        limit=SEARCH_RESULT_LIMIT,
    )
    if not results:
        return ToolOutcome(
            data={"status": "NO_RESULTS", "camera": camera},
            reply=text(ctx.language, "no_results"),
        )

    best = results[0].product
    if not best.in_stock:
        alternatives = await gateway.alternatives_for(best.product_id, lab)
        outcome = _sold_out_outcome(best, alternatives, ctx)
        outcome.data["camera"] = camera
        return outcome

    return ToolOutcome(
        data={
            "status": "OK",
            "camera": camera,
            "results": [_product_summary(r) for r in results],
        },
        products=[r.product.product_id for r in results],
    )


async def get_product_details(
    args: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
) -> ToolOutcome:
    product, error = await _resolve_product(_arg_str(args, "product_id"), ctx, gateway)
    if product is None:
        return _error_outcome(error or "not_found", ctx)
    return ToolOutcome(
        data={
            "status": "OK",
            "product_id": product.product_id,
            "name": product.name,
            "description": product.description,
            "price": product.base_price,
            "color": product.color_name,
            "in_stock": product.in_stock,
            "sizes": [
                {"size": v.size, "in_stock": v.stock > 0, "variant_id": v.variant_id}
                for v in product.variants
            ],
        },
        products=[product.product_id],
    )


def _navigation_commands(waypoint_id: str, product_id: str | None) -> list[Any]:
    commands: list[Any] = [
        c.PlayAvatarAnimationCommand(
            action="PLAY_AVATAR_ANIMATION",
            payload=c.PlayAvatarAnimationPayload(animation="GESTURE_GUIDE"),
        ),
        c.NavigateToWaypointCommand(
            action="NAVIGATE_TO_WAYPOINT",
            payload=c.NavigateToWaypointPayload(waypoint_id=waypoint_id),
        ),
    ]
    if product_id is not None:
        commands.append(
            c.HighlightProductsCommand(
                action="HIGHLIGHT_PRODUCTS",
                payload=c.HighlightProductsPayload(product_ids=[product_id]),
            )
        )
    return commands


async def navigate_to_product(
    args: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
) -> ToolOutcome:
    language = ctx.language
    product_raw = _arg_str(args, "product_id")
    category_raw = _arg_str(args, "category")

    if product_raw is None and category_raw is not None:
        slug = await _resolve_category(category_raw, gateway)
        categories = await gateway.list_categories()
        target = next((cat for cat in categories if cat.slug == slug), None)
        if target is None or target.waypoint_id is None:
            names = join_options([cat.name for cat in categories], language)
            return _error_outcome("unknown_category", ctx, categories=names)
        return ToolOutcome(
            data={"status": "NAVIGATING", "category": target.slug},
            reply=text(
                language,
                "navigate_category",
                category=_CATEGORY_WORDS[language].get(target.slug, target.name),
            ),
            commands=_navigation_commands(target.waypoint_id, None),
            animation="GESTURE_GUIDE",
        )

    product, error = await _resolve_product(product_raw, ctx, gateway)
    if product is None:
        return _error_outcome(error or "not_found", ctx)
    return ToolOutcome(
        data={
            "status": "NAVIGATING",
            "product_id": product.product_id,
            "waypoint_id": product.waypoint_id,
        },
        reply=text(language, "navigate_product", name=product.name),
        commands=_navigation_commands(product.waypoint_id, product.product_id),
        products=[product.product_id],
        animation="GESTURE_GUIDE",
    )


def _match_variant(
    product: ProductInfo, variant_id: str | None, size: str | None
) -> VariantInfo | None:
    if variant_id:
        return next((v for v in product.variants if v.variant_id == variant_id), None)
    if size:
        wanted = size.strip().upper().removeprefix("SIZE").strip()
        return next((v for v in product.variants if v.size.upper() == wanted), None)
    if len(product.variants) == 1:
        return product.variants[0]
    return None


async def add_to_cart(
    args: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
) -> ToolOutcome:
    language = ctx.language
    product, error = await _resolve_product(_arg_str(args, "product_id"), ctx, gateway)
    if product is None:
        return _error_outcome(error or "not_found", ctx)

    quantity = _arg_int(args, "quantity", default=1, minimum=1, maximum=MAX_QUANTITY)
    in_stock_sizes = [v.size for v in product.in_stock_variants]
    if not in_stock_sizes:
        alternatives = await gateway.alternatives_for(product.product_id, None)
        return _sold_out_outcome(product, alternatives, ctx)

    variant = _match_variant(
        product, _arg_str(args, "variant_id"), _arg_str(args, "size")
    )
    if variant is None:
        # Never guess a size: ask, offering only the sizes that are in stock.
        return ToolOutcome(
            data={
                "status": "NEEDS_VARIANT",
                "product_id": product.product_id,
                "available_sizes": in_stock_sizes,
            },
            reply=text(
                language,
                "needs_variant",
                name=product.name,
                options=join_options(in_stock_sizes, language),
            ),
            products=[product.product_id],
        )

    if variant.stock <= 0:
        return ToolOutcome(
            data={
                "status": "SIZE_SOLD_OUT",
                "product_id": product.product_id,
                "size": variant.size,
                "available_sizes": in_stock_sizes,
            },
            reply=text(
                language,
                "size_sold_out",
                name=product.name,
                size=variant.size,
                options=join_options(in_stock_sizes, language),
            ),
            products=[product.product_id],
            animation="SHRUG",
        )

    try:
        cart = await gateway.add_to_cart(variant.variant_id, quantity)
    except CartError as exc:
        if exc.code != "STOCK_INSUFFICIENT":
            raise
        available = (exc.details or {}).get("available", variant.stock)
        return ToolOutcome(
            data={
                "status": "STOCK_INSUFFICIENT",
                "product_id": product.product_id,
                "available": available,
            },
            reply=text(
                language,
                "stock_limited",
                name=product.name,
                size=variant.size,
                available=available,
            ),
            products=[product.product_id],
            animation="SHRUG",
        )

    return ToolOutcome(
        data={
            "status": "ADDED",
            "product_id": product.product_id,
            "variant_id": variant.variant_id,
            "quantity": quantity,
            "cart_item_count": cart.item_count,
            "cart_total": cart.total_amount,
        },
        reply=text(
            language,
            "added",
            quantity=quantity,
            name=product.name,
            size=variant.size,
            count=cart.item_count,
            total=format_price(cart.total_amount, language),
        ),
        cart=cart,
        products=[product.product_id],
        animation="NOD",
    )


async def open_product_inspector(
    args: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
) -> ToolOutcome:
    product, error = await _resolve_product(_arg_str(args, "product_id"), ctx, gateway)
    if product is None:
        return _error_outcome(error or "not_found", ctx)
    return ToolOutcome(
        data={"status": "OPENED", "product_id": product.product_id},
        reply=text(ctx.language, "inspector", name=product.name),
        ui_action=c.OpenProductInspectorAction(
            action="OPEN_PRODUCT_INSPECTOR", product_id=product.product_id
        ),
        products=[product.product_id],
        animation="NOD",
    )


async def start_checkout(
    args: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
) -> ToolOutcome:
    cart = await gateway.cart_summary()
    if cart.item_count == 0:
        return ToolOutcome(
            data={"status": "CART_EMPTY"},
            reply=text(ctx.language, "checkout_empty"),
        )
    return ToolOutcome(
        data={"status": "OPENED", "cart_total": cart.total_amount},
        reply=text(
            ctx.language,
            "checkout",
            count=cart.item_count,
            total=format_price(cart.total_amount, ctx.language),
        ),
        ui_action=c.OpenCheckoutAction(action="OPEN_CHECKOUT"),
        animation="NOD",
    )


ToolFunction = Any  # async (args, ctx, gateway) -> ToolOutcome

TOOLS: dict[str, ToolFunction] = {
    "search_inventory": search_inventory,
    "get_product_details": get_product_details,
    "navigate_to_product": navigate_to_product,
    "add_to_cart": add_to_cart,
    "open_product_inspector": open_product_inspector,
    "start_checkout": start_checkout,
}


async def run_tool(
    name: str, args: dict[str, Any], ctx: ToolContext, gateway: StoreGateway
) -> ToolOutcome:
    """Run one tool. Unknown names and unexpected failures become error outcomes, so a
    bad model output can never crash the conversation."""
    function = TOOLS.get(name)
    if function is None:
        logger.warning("Model called unknown tool %r", name)
        return ToolOutcome(data={"status": "ERROR", "error": f"unknown tool {name}"})
    try:
        outcome: ToolOutcome = await function(args, ctx, gateway)
        return outcome
    except Exception:
        logger.exception("Tool %s failed", name)
        return _error_outcome("tool_failed", ctx)
