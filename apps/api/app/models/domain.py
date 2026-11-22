from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

CART_STATUS_ACTIVE = "ACTIVE"
CART_STATUS_CHECKED_OUT = "CHECKED_OUT"
ORDER_STATUS_CONFIRMED = "CONFIRMED"
PAYMENT_METHOD_SIMULATED = "SIMULATED"

IMAGE_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 1536

# Prices are whole currency units (VND), matching the integer prices in the API.
Money = Numeric(12, 0)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "is_guest OR (email IS NOT NULL AND password_hash IS NOT NULL)",
            name="ck_users_registered_has_credentials",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    is_guest: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    accessibility_prefs: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    aisle_zone_code: Mapped[str] = mapped_column(String(20))


class ShelfLocation(Base):
    __tablename__ = "shelf_locations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    aisle_code: Mapped[str] = mapped_column(String(20))
    pos_x: Mapped[float] = mapped_column(Float)
    pos_y: Mapped[float] = mapped_column(Float)
    pos_z: Mapped[float] = mapped_column(Float)
    # Exposed as `waypoint_id` everywhere outside the database.
    unity_waypoint_id: Mapped[str] = mapped_column(String(64), unique=True)


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        Index(
            "ix_products_image_embedding_hnsw",
            "image_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"image_embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Public string ID used by API, WebSocket and the Unity bridge (e.g. "prod_001").
    public_id: Mapped[str] = mapped_column(String(32), unique=True)
    sku: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    category_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("categories.id"))
    shelf_location_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("shelf_locations.id")
    )
    base_price: Mapped[Decimal] = mapped_column(Money)
    primary_color_hex: Mapped[str] = mapped_column(String(7))
    color_lab: Mapped[list[float]] = mapped_column(ARRAY(Float))
    image_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(IMAGE_EMBEDDING_DIM)
    )
    # Reserved for text semantic search; stays NULL unless that feature is built.
    text_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(TEXT_EMBEDDING_DIM)
    )
    unity_prefab_id: Mapped[str] = mapped_column(String(100))
    glb_model_url: Mapped[str | None] = mapped_column(String(500))

    category: Mapped[Category] = relationship(lazy="selectin")
    shelf_location: Mapped[ShelfLocation] = relationship(lazy="selectin")
    variants: Mapped[list[ProductVariant]] = relationship(
        back_populates="product",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="ProductVariant.public_id",
    )


class ProductVariant(Base):
    __tablename__ = "product_variants"
    __table_args__ = (
        UniqueConstraint("product_id", "size", "color_name", name="uq_variant_option"),
        CheckConstraint("stock_qty >= 0", name="ck_variant_stock_non_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Public string ID, e.g. "var_001_L".
    public_id: Mapped[str] = mapped_column(String(48), unique=True)
    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE")
    )
    size: Mapped[str] = mapped_column(String(20))
    color_name: Mapped[str] = mapped_column(String(50))
    stock_qty: Mapped[int] = mapped_column(Integer, default=0)
    price_delta: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))

    product: Mapped[Product] = relationship(back_populates="variants")


class Cart(Base):
    __tablename__ = "carts"
    __table_args__ = (
        CheckConstraint(
            f"status IN ('{CART_STATUS_ACTIVE}', '{CART_STATUS_CHECKED_OUT}')",
            name="ck_cart_status",
        ),
        # At most one ACTIVE cart per user.
        Index(
            "uq_carts_one_active_per_user",
            "user_id",
            unique=True,
            postgresql_where=text(f"status = '{CART_STATUS_ACTIVE}'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(20), default=CART_STATUS_ACTIVE)
    # Optimistic lock: services bump it on every change; clients send it as If-Match.
    version: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    items: Mapped[list[CartItem]] = relationship(
        back_populates="cart", lazy="selectin", cascade="all, delete-orphan"
    )


class CartItem(Base):
    __tablename__ = "cart_items"
    __table_args__ = (
        UniqueConstraint("cart_id", "variant_id", name="uq_cart_variant"),
        CheckConstraint("quantity > 0", name="ck_cart_item_quantity_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    cart_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("carts.id", ondelete="CASCADE")
    )
    variant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("product_variants.id"))
    quantity: Mapped[int] = mapped_column(Integer)

    cart: Mapped[Cart] = relationship(back_populates="items")
    variant: Mapped[ProductVariant] = relationship(lazy="selectin")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Public string ID, e.g. "ord_0001".
    public_id: Mapped[str] = mapped_column(String(32), unique=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    # A cart converts to at most one order.
    cart_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("carts.id"), unique=True)
    total_amount: Mapped[Decimal] = mapped_column(Money)
    shipping_info: Mapped[dict[str, Any]] = mapped_column(JSONB)
    payment_method: Mapped[str] = mapped_column(
        String(20), default=PAYMENT_METHOD_SIMULATED
    )
    status: Mapped[str] = mapped_column(String(20), default=ORDER_STATUS_CONFIRMED)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order", lazy="selectin", cascade="all, delete-orphan"
    )


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_order_item_quantity_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    order_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE")
    )
    variant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("product_variants.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    # Copied from the catalogue at checkout time.
    unit_price: Mapped[Decimal] = mapped_column(Money)

    order: Mapped[Order] = relationship(back_populates="items")
    variant: Mapped[ProductVariant] = relationship(lazy="selectin")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    context_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE")
    )
    modality: Mapped[str] = mapped_column(String(20))
    user_transcript: Mapped[str | None] = mapped_column(Text)
    # Extracted features only (colour, label, confidence); raw images are never stored.
    cv_features: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    agent_response: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
