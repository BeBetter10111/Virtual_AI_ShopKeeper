import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.domain import (
    CART_STATUS_ACTIVE,
    Cart,
    CartItem,
    ProductVariant,
)
from app.schemas import contracts as c

_NAMED_SIZES = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}


class CartError(Exception):
    """Cart-related failure that maps to a standard ApiError response."""

    def __init__(
        self,
        code: c.ApiErrorCode,
        status_code: int,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message
        self.details = details


def not_found(message: str) -> CartError:
    return CartError("NOT_FOUND", 404, message)


def size_sort_key(size: str) -> tuple[int, float, str]:
    """Order sizes as XS < S < M < L < XL < XXL, then numeric sizes, then the rest."""
    upper = size.upper()
    if upper in _NAMED_SIZES:
        return (0, float(_NAMED_SIZES[upper]), "")
    try:
        return (1, float(size), "")
    except ValueError:
        return (2, 0.0, size)


@asynccontextmanager
async def _rollback_on_error(db: AsyncSession) -> AsyncIterator[None]:
    """Roll back at once on failure so the cart row lock is released."""
    try:
        yield
    except BaseException:
        await db.rollback()
        raise


# Loading


def _cart_query(user_id: uuid.UUID) -> Select[tuple[Cart]]:
    return (
        select(Cart)
        .where(Cart.user_id == user_id, Cart.status == CART_STATUS_ACTIVE)
        .options(
            selectinload(Cart.items)
            .selectinload(CartItem.variant)
            .selectinload(ProductVariant.product)
        )
        .execution_options(populate_existing=True)
    )


async def _active_cart(db: AsyncSession, user_id: uuid.UUID, lock: bool) -> Cart:
    """Return the user's ACTIVE cart, creating it if needed.

    With lock=True the cart row is locked (SELECT ... FOR UPDATE) until commit,
    so concurrent changes to one cart are applied one after another.
    """
    query = _cart_query(user_id)
    if lock:
        query = query.with_for_update()
    cart: Cart | None = (await db.execute(query)).scalar_one_or_none()
    if cart is not None:
        return cart

    try:
        async with db.begin_nested():
            db.add(Cart(user_id=user_id, status=CART_STATUS_ACTIVE, version=0))
            await db.flush()
    except IntegrityError:
        pass  # Another request created it first; load that one.
    cart = (await db.execute(query)).scalar_one()
    return cart


async def _get_variant(db: AsyncSession, public_id: str) -> ProductVariant:
    variant = (
        await db.execute(
            select(ProductVariant)
            .where(ProductVariant.public_id == public_id)
            .options(selectinload(ProductVariant.product))
        )
    ).scalar_one_or_none()
    if variant is None:
        raise not_found("Product variant not found.")
    return variant


def _find_item(cart: Cart, variant_public_id: str) -> CartItem:
    for item in cart.items:
        if item.variant.public_id == variant_public_id:
            return item
    raise not_found("Item is not in the cart.")


# Checks 


def _check_version(cart: Cart, expected: int | None) -> None:
    if expected is not None and expected != cart.version:
        raise CartError(
            "VERSION_CONFLICT",
            409,
            "The cart was changed by another request. Reload it and retry.",
            {"current_version": cart.version},
        )


def _check_stock(variant: ProductVariant, wanted: int) -> None:
    if wanted > variant.stock_qty:
        raise CartError(
            "STOCK_INSUFFICIENT",
            409,
            f"Only {variant.stock_qty} left in stock.",
            {"available": variant.stock_qty, "requested": wanted},
        )


# Summary 


def build_cart_summary(cart: Cart) -> c.CartSummary:
    """Convert a fully loaded Cart (items -> variant -> product) to the DTO."""
    rows = sorted(
        cart.items,
        key=lambda i: (i.variant.product.public_id, size_sort_key(i.variant.size)),
    )
    items = [
        c.CartItem(
            product_id=row.variant.product.public_id,
            variant_id=row.variant.public_id,
            name=row.variant.product.name,
            size=row.variant.size,
            quantity=row.quantity,
            unit_price=int(row.variant.product.base_price + row.variant.price_delta),
        )
        for row in rows
    ]
    return c.CartSummary(
        cart_id=str(cart.id),
        version=cart.version,
        items=items,
        item_count=sum(i.quantity for i in items),
        total_amount=sum(i.quantity * i.unit_price for i in items),
    )


async def get_cart_summary(db: AsyncSession, user_id: uuid.UUID) -> c.CartSummary:
    cart = await _active_cart(db, user_id, lock=False)
    summary = build_cart_summary(cart)
    await db.commit()  # persists a newly created empty cart
    return summary


async def _commit_and_summarize(
    db: AsyncSession, user_id: uuid.UUID
) -> c.CartSummary:
    await db.commit()
    cart = (await db.execute(_cart_query(user_id))).scalar_one()
    return build_cart_summary(cart)


# Operations


async def add_item(
    db: AsyncSession,
    user_id: uuid.UUID,
    variant_public_id: str,
    quantity: int,
    expected_version: int | None,
) -> c.CartSummary:
    async with _rollback_on_error(db):
        cart = await _active_cart(db, user_id, lock=True)
        _check_version(cart, expected_version)
        variant = await _get_variant(db, variant_public_id)

        existing = next((i for i in cart.items if i.variant_id == variant.id), None)
        new_quantity = (existing.quantity if existing else 0) + quantity
        _check_stock(variant, new_quantity)

        if existing is not None:
            existing.quantity = new_quantity
        else:
            cart.items.append(CartItem(variant_id=variant.id, quantity=new_quantity))
        cart.version += 1
        return await _commit_and_summarize(db, user_id)


async def set_item_quantity(
    db: AsyncSession,
    user_id: uuid.UUID,
    variant_public_id: str,
    quantity: int,
    expected_version: int | None,
) -> c.CartSummary:
    async with _rollback_on_error(db):
        cart = await _active_cart(db, user_id, lock=True)
        _check_version(cart, expected_version)
        item = _find_item(cart, variant_public_id)
        _check_stock(item.variant, quantity)

        item.quantity = quantity
        cart.version += 1
        return await _commit_and_summarize(db, user_id)


async def remove_item(
    db: AsyncSession,
    user_id: uuid.UUID,
    variant_public_id: str,
    expected_version: int | None,
) -> c.CartSummary:
    async with _rollback_on_error(db):
        cart = await _active_cart(db, user_id, lock=True)
        _check_version(cart, expected_version)
        item = _find_item(cart, variant_public_id)

        cart.items.remove(item)
        cart.version += 1
        return await _commit_and_summarize(db, user_id)
