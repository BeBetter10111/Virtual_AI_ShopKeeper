"""Product search: colour distance (CIEDE2000) blended with pgvector similarity.

`rank_products` is the general search used by the visual-search REST/WebSocket paths
and by the agent's search_inventory tool. `visual_search` turns a `VisionResult` into
the `VisualSearchResponse` contract, including in-stock alternatives.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Float, Select, null, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import Category, Product
from app.schemas import contracts as c
from app.services.vision_service import VisionResult, delta_e_2000

MAX_MATCHES = 5
MAX_ALTERNATIVES = 2
ALTERNATIVE_MAX_DELTA_E = 10.0  # alternatives must be strictly closer than this
CANDIDATE_LIMIT = 50  # rows fetched from the database before re-ranking

# Final score = COLOR_WEIGHT * colour + EMBEDDING_WEIGHT * embedding similarity.
# Colour is what the shopper asks for ("this colour"), so it counts more. These
# weights are untuned starting values.
COLOR_WEIGHT = 0.7
EMBEDDING_WEIGHT = 0.3
COLOR_SCALE_DELTA_E = 50.0  # a colour difference of this size scores 0


@dataclass(frozen=True)
class ScoredProduct:
    product: Product
    delta_e00: float | None  # None when no colour was given
    similarity: float  # cosine similarity of image embeddings, 0..1
    score: float  # combined score, 0..1
    in_stock: bool


def color_score(delta_e: float) -> float:
    return max(0.0, 1.0 - delta_e / COLOR_SCALE_DELTA_E)


def combine_score(
    delta_e: float | None, similarity: float, use_embedding: bool
) -> float:
    """Blend colour and embedding similarity into one 0..1 score."""
    if delta_e is None:
        return similarity if use_embedding else 0.0
    if not use_embedding:
        return color_score(delta_e)
    return COLOR_WEIGHT * color_score(delta_e) + EMBEDDING_WEIGHT * similarity


def is_in_stock(product: Product) -> bool:
    """A product is in stock when at least one of its variants is."""
    return any(variant.stock_qty > 0 for variant in product.variants)


def pick_alternatives(
    ranked: Sequence[ScoredProduct], best: ScoredProduct
) -> list[ScoredProduct]:
    """In-stock products near the shopper's colour, offered when `best` is sold out.

    `ranked` must already be limited to the category of `best`. Closeness is measured
    against the colour the shopper showed, not against the sold-out product.
    """
    if best.in_stock:
        return []
    candidates = [
        s
        for s in ranked
        if s.in_stock
        and s.delta_e00 is not None
        and s.delta_e00 < ALTERNATIVE_MAX_DELTA_E
        and s.product.id != best.product.id
    ]
    candidates.sort(key=lambda s: (s.delta_e00 or 0.0, s.product.public_id))
    return candidates[:MAX_ALTERNATIVES]


async def rank_products(
    db: AsyncSession,
    *,
    query_text: str | None = None,
    lab: Sequence[float] | None = None,
    embedding: Sequence[float] | None = None,
    category_slug: str | None = None,
) -> list[ScoredProduct]:
    """Score products and return them best-first (at most CANDIDATE_LIMIT).

    Filters: `category_slug`, and `query_text` (every word must appear in the name or
    description, case-insensitive). Scoring uses `lab` (colour) and `embedding`
    (pgvector cosine similarity) when given.
    """
    statement: Select[Any]
    if embedding is not None:
        distance = Product.image_embedding.op("<=>", return_type=Float)(list(embedding))
        statement = select(Product, distance.label("distance")).order_by(distance)
    else:
        statement = select(Product, null().label("distance"))
        statement = statement.order_by(Product.public_id)

    if category_slug is not None:
        statement = statement.join(Category, Product.category_id == Category.id).where(
            Category.slug == category_slug
        )
    for word in (query_text or "").split():
        statement = statement.where(
            Product.name.icontains(word, autoescape=True)
            | Product.description.icontains(word, autoescape=True)
        )

    rows = (await db.execute(statement.limit(CANDIDATE_LIMIT))).all()

    use_embedding = embedding is not None
    scored: list[ScoredProduct] = []
    for product, distance_value in rows:
        similarity = (
            0.0
            if distance_value is None
            else min(1.0, max(0.0, 1.0 - float(distance_value)))
        )
        delta_e = delta_e_2000(lab, product.color_lab) if lab is not None else None
        scored.append(
            ScoredProduct(
                product=product,
                delta_e00=delta_e,
                similarity=similarity,
                score=combine_score(delta_e, similarity, use_embedding),
                in_stock=is_in_stock(product),
            )
        )
    scored.sort(key=lambda s: (-s.score, s.delta_e00 or 0.0, s.product.public_id))
    return scored


def to_match(scored: ScoredProduct) -> c.VisualSearchMatch:
    return c.VisualSearchMatch(
        product_id=scored.product.public_id,
        name=scored.product.name,
        similarity_score=round(min(1.0, max(0.0, scored.score)), 3),
        delta_e00=round(scored.delta_e00 or 0.0, 2),
        waypoint_id=scored.product.shelf_location.unity_waypoint_id,
        in_stock=scored.in_stock,
    )


async def visual_search(
    db: AsyncSession, analysis: VisionResult
) -> c.VisualSearchResponse:
    """Find store products matching an analysed webcam frame.

    Out-of-domain objects get no matches. Otherwise up to MAX_MATCHES products of
    the detected category are returned, and `alternatives` is filled only when the
    best match is sold out.
    """
    matches: list[ScoredProduct] = []
    alternatives: list[ScoredProduct] = []
    if analysis.is_in_domain and analysis.category_slug is not None:
        ranked = await rank_products(
            db,
            lab=analysis.lab,
            embedding=analysis.embedding,
            category_slug=analysis.category_slug,
        )
        matches = ranked[:MAX_MATCHES]
        if matches:
            alternatives = pick_alternatives(ranked, matches[0])

    return c.VisualSearchResponse(
        detected_color_name=analysis.color_name,
        detected_lab=[round(v, 2) for v in analysis.lab],
        category_label=analysis.category_label,
        category_confidence=round(analysis.category_confidence, 3),
        is_in_domain=analysis.is_in_domain,
        quality_warning=analysis.quality_warning,
        matches=[to_match(s) for s in matches],
        alternatives=[to_match(s) for s in alternatives],
    )
