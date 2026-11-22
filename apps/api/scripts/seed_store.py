"""Create the tables and load sample store data.

Run from apps/api:
    python -m scripts.seed_store [--reset] [--database-url URL]

IMPORTANT: every `image_embedding` written here is SYNTHETIC SEED DATA (a
deterministic random unit vector per product), not the output of a CLIP model.
Visual search by embedding is therefore not meaningful until Task 3.1 replaces
these vectors with real embeddings. Colours (`color_lab`) are real: they are
computed from each product's hex colour.
"""

import argparse
import asyncio
import math
import random
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import get_settings
from app.core.database import (
    Base,
    create_db_engine,
    create_session_factory,
    ensure_pgvector,
)
from app.models.domain import (
    IMAGE_EMBEDDING_DIM,
    Category,
    Product,
    ProductVariant,
    ShelfLocation,
)

CHECKOUT_WAYPOINT_ID = "WP_CHECKOUT"

# slug -> (display name, aisle zone code)
CATEGORIES: dict[str, tuple[str, str]] = {
    "shirts": ("Shirts", "A1"),
    "jackets": ("Jackets", "A2"),
    "pants": ("Pants", "A3"),
    "shoes": ("Shoes", "A4"),
}

# aisle code -> (waypoint_id, x, y, z). Placeholder coordinates until the Unity
# scene (Task 4.5) is final.
SHELVES: dict[str, tuple[str, float, float, float]] = {
    "A1": ("WP_AISLE_01_SHIRTS", 12.5, 0.0, -4.2),
    "A2": ("WP_AISLE_02_JACKETS", 12.5, 0.0, -9.0),
    "A3": ("WP_AISLE_03_PANTS", 4.0, 0.0, -12.0),
    "A4": ("WP_AISLE_04_SHOES", -4.0, 0.0, -12.0),
}

SHIRT_SIZES = ("S", "M", "L", "XL")
SHOE_SIZES = ("40", "41", "42", "43")


@dataclass(frozen=True)
class ProductSeed:
    number: int
    name: str
    description: str
    category_slug: str
    color_name: str
    color_hex: str
    price: int
    sizes: tuple[str, ...]
    stock: tuple[int, ...]

    @property
    def public_id(self) -> str:
        return f"prod_{self.number:03d}"


PRODUCTS: tuple[ProductSeed, ...] = (
    ProductSeed(1, "Blue Oxford Shirt", "Classic button-down Oxford shirt.",
                "shirts", "Navy Blue", "#1F3A5F", 450000, SHIRT_SIZES, (8, 5, 0, 3)),
    ProductSeed(2, "White Linen Shirt", "Lightweight linen shirt for warm days.",
                "shirts", "White", "#F2F2EE", 520000, SHIRT_SIZES, (6, 9, 7, 2)),
    # Fully out of stock on purpose: exercises the "suggest alternatives" flow.
    ProductSeed(3, "Red Flannel Shirt", "Soft brushed-cotton flannel shirt.",
                "shirts", "Red", "#B22222", 480000, SHIRT_SIZES, (0, 0, 0, 0)),
    ProductSeed(4, "Brick Red Cotton Shirt", "Everyday cotton shirt in brick red.",
                "shirts", "Brick Red", "#A83232", 430000, SHIRT_SIZES, (4, 6, 5, 3)),
    ProductSeed(5, "Black Leather Jacket", "Genuine leather biker jacket.",
                "jackets", "Black", "#1A1A1A", 2490000, SHIRT_SIZES, (2, 3, 2, 1)),
    ProductSeed(6, "Denim Jacket", "Mid-wash denim trucker jacket.",
                "jackets", "Denim Blue", "#4F6D8F", 990000, SHIRT_SIZES, (5, 7, 4, 2)),
    ProductSeed(7, "Beige Trench Coat", "Water-resistant double-breasted trench.",
                "jackets", "Beige", "#C8B79A", 1890000, SHIRT_SIZES, (3, 4, 3, 0)),
    ProductSeed(8, "Khaki Chinos", "Straight-fit stretch chinos.",
                "pants", "Khaki", "#B5A27A", 620000, SHIRT_SIZES, (7, 8, 6, 4)),
    ProductSeed(9, "Black Slim Jeans", "Slim-fit black stretch denim.",
                "pants", "Black", "#222831", 750000, SHIRT_SIZES, (5, 6, 5, 2)),
    ProductSeed(10, "Grey Jogger Pants", "Tapered fleece joggers.",
                "pants", "Grey", "#808080", 480000, SHIRT_SIZES, (9, 10, 8, 5)),
    ProductSeed(11, "White Sneakers", "Minimal low-top leather sneakers.",
                "shoes", "White", "#F5F5F5", 1250000, SHOE_SIZES, (4, 6, 5, 2)),
    ProductSeed(12, "Brown Leather Boots", "Lace-up leather boots.",
                "shoes", "Brown", "#5C3A21", 1790000, SHOE_SIZES, (3, 0, 2, 1)),
)


def hex_to_lab(hex_color: str) -> list[float]:
    """Convert '#RRGGBB' (sRGB, D65) to CIELAB [L, a, b], rounded to 2 decimals."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = linear(r), linear(g), linear(b)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    lab = (116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))
    # "+ 0.0" turns -0.0 into 0.0.
    return [round(v, 2) + 0.0 for v in lab]


def make_seed_embedding(key: str) -> list[float]:
    """Deterministic synthetic unit vector. NOT a CLIP embedding."""
    rng = random.Random(key)
    vec = [rng.gauss(0.0, 1.0) for _ in range(IMAGE_EMBEDDING_DIM)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def prefab_id(name: str) -> str:
    return "Pref_" + "".join(part.capitalize() for part in name.split())


async def seed(engine: AsyncEngine, reset: bool = False) -> int:
    """Create tables and insert sample data. Returns the number of products added."""
    await ensure_pgvector(engine)
    async with engine.begin() as conn:
        if reset:
            await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    factory = create_session_factory(engine)
    async with factory() as session:
        existing = await session.scalar(select(func.count()).select_from(Product))
        if existing:
            return 0

        categories = {
            slug: Category(name=name, slug=slug, aisle_zone_code=zone)
            for slug, (name, zone) in CATEGORIES.items()
        }
        shelves = {
            aisle: ShelfLocation(
                aisle_code=aisle, pos_x=x, pos_y=y, pos_z=z, unity_waypoint_id=wp
            )
            for aisle, (wp, x, y, z) in SHELVES.items()
        }
        session.add_all([*categories.values(), *shelves.values()])

        for spec in PRODUCTS:
            category = categories[spec.category_slug]
            product = Product(
                public_id=spec.public_id,
                sku=f"SKU-{spec.number:03d}",
                name=spec.name,
                description=spec.description,
                category=category,
                shelf_location=shelves[category.aisle_zone_code],
                base_price=Decimal(spec.price),
                primary_color_hex=spec.color_hex,
                color_lab=hex_to_lab(spec.color_hex),
                image_embedding=make_seed_embedding(spec.public_id),
                unity_prefab_id=prefab_id(spec.name),
                glb_model_url=None,
                variants=[
                    ProductVariant(
                        public_id=f"var_{spec.number:03d}_{size}",
                        size=size,
                        color_name=spec.color_name,
                        stock_qty=qty,
                        price_delta=Decimal(0),
                    )
                    for size, qty in zip(spec.sizes, spec.stock, strict=True)
                ],
            )
            session.add(product)
        await session.commit()
    return len(PRODUCTS)


async def _main(database_url: str, reset: bool) -> None:
    engine = create_db_engine(database_url)
    try:
        added = await seed(engine, reset=reset)
    finally:
        await engine.dispose()
    if added:
        print(f"Seeded {added} products (embeddings are SYNTHETIC seed data).")
    else:
        print("Products already present; nothing to do. Use --reset to rebuild.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tables and seed store data.")
    parser.add_argument("--reset", action="store_true", help="drop all tables first")
    parser.add_argument("--database-url", default=get_settings().database_url)
    args = parser.parse_args()
    asyncio.run(_main(args.database_url, args.reset))


if __name__ == "__main__":
    main()
