import uuid
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload

from app.core.config import Settings, get_settings
from app.core.database import get_db_session
from app.models.domain import Product, ShelfLocation
from app.schemas import contracts as c
from app.services import auth_service, cart_service
from app.services.auth_service import AuthError, RegistrationError
from app.services.cart_service import CartError, size_sort_key

CHECKOUT_WAYPOINT_ID = "WP_CHECKOUT"

SERVICE_ERRORS = (AuthError, RegistrationError, CartError)


def error_response(
    status_code: int,
    code: c.ApiErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = c.ApiError(error=c.ApiErrorBody(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


class ApiRoute(APIRoute):
    """Turn service errors and request-validation errors into ApiError responses."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError as exc:
                errors = [
                    {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                    for e in exc.errors()
                ]
                return error_response(
                    422,
                    "VALIDATION_ERROR",
                    "Request validation failed.",
                    {"errors": errors},
                )
            except SERVICE_ERRORS as exc:
                is_auth = isinstance(exc, AuthError)
                headers = {"WWW-Authenticate": "Bearer"} if is_auth else None
                return error_response(
                    exc.status_code, exc.code, exc.message, exc.details, headers
                )

        return handler


router = APIRouter(prefix="/api/v1", route_class=ApiRoute)

bearer_scheme = HTTPBearer(auto_error=False)
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
AppSettings = Annotated[Settings, Depends(get_settings)]


@dataclass(frozen=True)
class AuthContext:
    user_id: uuid.UUID
    session_id: str


async def get_auth_context(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: DbSession,
    settings: AppSettings,
) -> AuthContext:
    if credentials is None:
        raise AuthError("Missing bearer token.")
    claims = auth_service.decode_access_token(credentials.credentials, settings)
    user = await auth_service.load_user(db, claims)
    return AuthContext(user_id=user.id, session_id=claims.session_id)


CurrentUser = Annotated[AuthContext, Depends(get_auth_context)]


def parse_if_match(value: str | None) -> int | None:
    """Accept `7` or `"7"` (ETag style); None when the header is absent."""
    if value is None:
        return None
    try:
        return int(value.strip().strip('"'))
    except ValueError:
        raise CartError(
            "VALIDATION_ERROR", 422, "If-Match must be the cart version number."
        ) from None


IfMatch = Annotated[str | None, Header(alias="If-Match")]


# --- Auth ---------------------------------------------------------------------


@router.post("/auth/session")
async def create_session(
    body: c.SessionRequest, db: DbSession, settings: AppSettings
) -> c.SessionResponse:
    if body.mode == "guest":
        return await auth_service.start_guest_session(
            db, body.accessibility_mode, settings
        )
    if body.email is None or body.password is None:
        raise AuthError("Email and password are required.")
    return await auth_service.login(db, body.email, body.password, settings)


@router.post("/auth/register")
async def register(
    body: c.RegisterRequest, db: DbSession, settings: AppSettings
) -> c.SessionResponse:
    return await auth_service.register_user(db, body, settings)


# Store 


@router.get("/store/layout")
async def get_store_layout(_: CurrentUser, db: DbSession) -> c.StoreLayout:
    shelf_rows = (
        (await db.execute(select(ShelfLocation).order_by(ShelfLocation.aisle_code)))
        .scalars()
        .all()
    )
    product_rows = (
        (
            await db.execute(
                select(Product)
                .options(noload(Product.variants))
                .order_by(Product.public_id)
            )
        )
        .scalars()
        .all()
    )

    by_shelf: dict[uuid.UUID, list[c.ShelfProduct]] = defaultdict(list)
    for product in product_rows:
        by_shelf[product.shelf_location_id].append(
            c.ShelfProduct(
                product_id=product.public_id,
                unity_prefab_id=product.unity_prefab_id,
                glb_model_url=product.glb_model_url,
                name=product.name,
                base_price=int(product.base_price),
            )
        )

    return c.StoreLayout(
        shelves=[
            c.Shelf(
                waypoint_id=shelf.unity_waypoint_id,
                aisle_code=shelf.aisle_code,
                position=c.Vector3(x=shelf.pos_x, y=shelf.pos_y, z=shelf.pos_z),
                products=by_shelf[shelf.id],
            )
            for shelf in shelf_rows
        ],
        checkout_waypoint_id=CHECKOUT_WAYPOINT_ID,
    )


@router.get("/products/{product_id}")
async def get_product(
    product_id: str, _: CurrentUser, db: DbSession
) -> c.ProductDetail:
    product = (
        await db.execute(select(Product).where(Product.public_id == product_id))
    ).scalar_one_or_none()
    if product is None:
        raise cart_service.not_found("Product not found.")

    variants = sorted(product.variants, key=lambda v: size_sort_key(v.size))
    return c.ProductDetail(
        product_id=product.public_id,
        name=product.name,
        description=product.description,
        base_price=int(product.base_price),
        glb_model_url=product.glb_model_url,
        waypoint_id=product.shelf_location.unity_waypoint_id,
        variants=[
            c.ProductVariant(
                variant_id=v.public_id,
                size=v.size,
                color_name=v.color_name,
                stock_qty=v.stock_qty,
                price_delta=int(v.price_delta),
            )
            for v in variants
        ],
    )


# --- Cart 


@router.get("/cart")
async def get_cart(user: CurrentUser, db: DbSession) -> c.CartSummary:
    return await cart_service.get_cart_summary(db, user.user_id)


@router.post("/cart/items")
async def add_cart_item(
    body: c.AddCartItemRequest,
    user: CurrentUser,
    db: DbSession,
    if_match: IfMatch = None,
) -> c.CartSummary:
    return await cart_service.add_item(
        db, user.user_id, body.variant_id, body.quantity, parse_if_match(if_match)
    )


@router.patch("/cart/items/{variant_id}")
async def update_cart_item(
    variant_id: str,
    body: c.UpdateCartItemRequest,
    user: CurrentUser,
    db: DbSession,
    if_match: IfMatch = None,
) -> c.CartSummary:
    return await cart_service.set_item_quantity(
        db, user.user_id, variant_id, body.quantity, parse_if_match(if_match)
    )


@router.delete("/cart/items/{variant_id}")
async def remove_cart_item(
    variant_id: str,
    user: CurrentUser,
    db: DbSession,
    if_match: IfMatch = None,
) -> c.CartSummary:
    return await cart_service.remove_item(
        db, user.user_id, variant_id, parse_if_match(if_match)
    )
