import asyncio
import uuid
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient, Response
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.core.config import get_settings
from app.models.domain import ProductVariant, User
from app.schemas import contracts as c
from app.services import auth_service
from scripts.seed_store import seed

BASE = "/api/v1"

# Seed data used below (see scripts/seed_store.py):
#   var_001_S stock 8, var_001_M stock 5 (450000), var_001_L stock 0, var_001_XL stock 3
#   var_002_M stock 9 (520000)


@pytest.fixture(autouse=True)
def fast_bcrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth_service, "BCRYPT_ROUNDS", 4)


@pytest_asyncio.fixture
async def seeded(db_engine: AsyncEngine) -> None:
    await seed(db_engine)


async def guest_headers(client: AsyncClient) -> dict[str, str]:
    response = await client.post(f"{BASE}/auth/session", json={"mode": "guest"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest_asyncio.fixture
async def auth(client: AsyncClient, seeded: None) -> dict[str, str]:
    return await guest_headers(client)


def assert_error(response: Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body = response.json()
    c.ApiError.model_validate(body)
    assert body["error"]["code"] == code
    error: dict[str, Any] = body["error"]
    return error


async def add_item(
    client: AsyncClient,
    headers: dict[str, str],
    variant_id: str,
    quantity: int = 1,
    **extra_headers: str,
) -> Response:
    return await client.post(
        f"{BASE}/cart/items",
        json={"variant_id": variant_id, "quantity": quantity},
        headers={**headers, **extra_headers},
    )


async def get_cart(client: AsyncClient, headers: dict[str, str]) -> c.CartSummary:
    response = await client.get(f"{BASE}/cart", headers=headers)
    assert response.status_code == 200, response.text
    return c.CartSummary.model_validate(response.json())


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_guest_session(client: AsyncClient, db_session: AsyncSession) -> None:
    response = await client.post(
        f"{BASE}/auth/session", json={"mode": "guest", "accessibility_mode": True}
    )
    assert response.status_code == 200
    session = c.SessionResponse.model_validate(response.json())
    assert session.expires_in == get_settings().access_token_ttl_seconds

    user = await db_session.get(User, uuid.UUID(session.user_id))
    assert user is not None
    assert user.is_guest
    assert user.accessibility_prefs == {"accessibility_mode": True}


async def test_register_stores_hashed_password_and_returns_usable_token(
    client: AsyncClient, db_session: AsyncSession, seeded: None
) -> None:
    password = "correct horse battery"
    response = await client.post(
        f"{BASE}/auth/register",
        json={"email": "Shopper@Example.com", "password": password},
    )
    assert response.status_code == 200
    session = c.SessionResponse.model_validate(response.json())

    result = await db_session.execute(
        select(User).where(User.email == "shopper@example.com")
    )
    user = result.scalar_one()
    assert not user.is_guest
    assert user.password_hash is not None
    assert user.password_hash.startswith("$2")
    assert password not in user.password_hash

    headers = {"Authorization": f"Bearer {session.access_token}"}
    assert (await client.get(f"{BASE}/cart", headers=headers)).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": "longenough"},
        {"email": "a@b.co", "password": "short"},
    ],
)
async def test_register_rejects_invalid_data(
    client: AsyncClient, payload: dict[str, str]
) -> None:
    response = await client.post(f"{BASE}/auth/register", json=payload)
    assert_error(response, 422, "VALIDATION_ERROR")


async def test_register_rejects_duplicate_email(client: AsyncClient) -> None:
    payload = {"email": "dup@example.com", "password": "longenough"}
    assert (await client.post(f"{BASE}/auth/register", json=payload)).status_code == 200
    again = {"email": "DUP@example.com", "password": "anotherlongone"}
    error = assert_error(
        await client.post(f"{BASE}/auth/register", json=again), 422, "VALIDATION_ERROR"
    )
    assert error["details"]["field"] == "email"


async def test_login_success_and_failures(client: AsyncClient) -> None:
    creds = {"email": "login@example.com", "password": "longenough"}
    registered = await client.post(f"{BASE}/auth/register", json=creds)
    user_id = registered.json()["user_id"]

    ok = await client.post(f"{BASE}/auth/session", json={"mode": "login", **creds})
    assert ok.status_code == 200
    assert ok.json()["user_id"] == user_id

    wrong = {"mode": "login", "email": creds["email"], "password": "wrong-password"}
    assert_error(
        await client.post(f"{BASE}/auth/session", json=wrong), 401, "UNAUTHORIZED"
    )

    unknown = {"mode": "login", "email": "nobody@example.com", "password": "longenough"}
    assert_error(
        await client.post(f"{BASE}/auth/session", json=unknown), 401, "UNAUTHORIZED"
    )


async def test_login_requires_credentials(client: AsyncClient) -> None:
    response = await client.post(f"{BASE}/auth/session", json={"mode": "login"})
    assert_error(response, 422, "VALIDATION_ERROR")


async def test_missing_token_is_unauthorized(client: AsyncClient, seeded: None) -> None:
    response = await client.get(f"{BASE}/cart")
    assert_error(response, 401, "UNAUTHORIZED")
    assert response.headers["www-authenticate"] == "Bearer"


async def test_malformed_and_wrong_scheme_tokens_are_unauthorized(
    client: AsyncClient, seeded: None
) -> None:
    for header in ("Bearer not-a-jwt", "Basic dXNlcjpwYXNz"):
        response = await client.get(f"{BASE}/cart", headers={"Authorization": header})
        assert_error(response, 401, "UNAUTHORIZED")


async def test_expired_token_is_unauthorized(client: AsyncClient, seeded: None) -> None:
    token = auth_service.create_access_token(
        uuid.uuid4(), "sid", get_settings(), ttl_seconds=-10
    )
    response = await client.get(
        f"{BASE}/cart", headers={"Authorization": f"Bearer {token}"}
    )
    assert_error(response, 401, "UNAUTHORIZED")


async def test_token_signed_with_other_secret_is_unauthorized(
    client: AsyncClient, seeded: None
) -> None:
    other = get_settings().model_copy(update={"jwt_secret_key": "some-other-secret"})
    token = auth_service.create_access_token(uuid.uuid4(), "sid", other)
    response = await client.get(
        f"{BASE}/cart", headers={"Authorization": f"Bearer {token}"}
    )
    assert_error(response, 401, "UNAUTHORIZED")


async def test_valid_token_for_unknown_user_is_unauthorized(
    client: AsyncClient, seeded: None
) -> None:
    token = auth_service.create_access_token(uuid.uuid4(), "sid", get_settings())
    response = await client.get(
        f"{BASE}/cart", headers={"Authorization": f"Bearer {token}"}
    )
    assert_error(response, 401, "UNAUTHORIZED")


async def test_every_protected_endpoint_requires_a_token(
    client: AsyncClient, seeded: None
) -> None:
    calls = [
        ("GET", "/store/layout", None),
        ("GET", "/products/prod_001", None),
        ("GET", "/cart", None),
        ("POST", "/cart/items", {"variant_id": "var_001_S", "quantity": 1}),
        ("PATCH", "/cart/items/var_001_S", {"quantity": 1}),
        ("DELETE", "/cart/items/var_001_S", None),
    ]
    for method, path, body in calls:
        response = await client.request(method, f"{BASE}{path}", json=body)
        assert_error(response, 401, "UNAUTHORIZED")


# ---------------------------------------------------------------------------
# Store layout and product detail
# ---------------------------------------------------------------------------


async def test_store_layout(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await client.get(f"{BASE}/store/layout", headers=auth)
    assert response.status_code == 200
    layout = c.StoreLayout.model_validate(response.json())

    assert layout.checkout_waypoint_id == "WP_CHECKOUT"
    assert [s.aisle_code for s in layout.shelves] == ["A1", "A2", "A3", "A4"]
    assert sum(len(s.products) for s in layout.shelves) == 12
    shirts = layout.shelves[0]
    assert shirts.waypoint_id == "WP_AISLE_01_SHIRTS"
    assert [p.product_id for p in shirts.products] == [
        "prod_001",
        "prod_002",
        "prod_003",
        "prod_004",
    ]
    assert shirts.products[0].unity_prefab_id == "Pref_BlueOxfordShirt"
    assert shirts.products[0].base_price == 450000


async def test_product_detail(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await client.get(f"{BASE}/products/prod_001", headers=auth)
    assert response.status_code == 200
    product = c.ProductDetail.model_validate(response.json())

    assert product.waypoint_id == "WP_AISLE_01_SHIRTS"
    assert [v.size for v in product.variants] == ["S", "M", "L", "XL"]
    assert {v.size: v.stock_qty for v in product.variants} == {
        "S": 8,
        "M": 5,
        "L": 0,
        "XL": 3,
    }

    shoes = await client.get(f"{BASE}/products/prod_011", headers=auth)
    assert [v["size"] for v in shoes.json()["variants"]] == ["40", "41", "42", "43"]


@pytest.mark.parametrize("product_id", ["prod_999", "abc", "PROD_001"])
async def test_unknown_product_is_not_found(
    client: AsyncClient, auth: dict[str, str], product_id: str
) -> None:
    response = await client.get(f"{BASE}/products/{product_id}", headers=auth)
    assert_error(response, 404, "NOT_FOUND")


# ---------------------------------------------------------------------------
# Cart
# ---------------------------------------------------------------------------


async def test_new_user_has_empty_cart(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    cart = await get_cart(client, auth)
    assert cart.version == 0
    assert cart.items == []
    assert cart.item_count == 0
    assert cart.total_amount == 0
    # Reading again must return the same cart, not create another one.
    assert (await get_cart(client, auth)).cart_id == cart.cart_id


async def test_add_item(client: AsyncClient, auth: dict[str, str]) -> None:
    response = await add_item(client, auth, "var_001_M", 2)
    assert response.status_code == 200
    cart = c.CartSummary.model_validate(response.json())

    assert cart.version == 1
    assert cart.item_count == 2
    assert cart.total_amount == 900000
    (item,) = cart.items
    assert (item.product_id, item.variant_id) == ("prod_001", "var_001_M")
    assert item.size == "M"
    assert (item.quantity, item.unit_price) == (2, 450000)


async def test_adding_same_variant_accumulates(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await add_item(client, auth, "var_001_M", 2)
    second = await add_item(client, auth, "var_001_M", 3)
    cart = c.CartSummary.model_validate(second.json())
    assert [i.quantity for i in cart.items] == [5]
    assert cart.version == 2

    over = await add_item(client, auth, "var_001_M", 1)
    error = assert_error(over, 409, "STOCK_INSUFFICIENT")
    assert error["details"]["available"] == 5


async def test_stock_insufficient_leaves_cart_unchanged(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await add_item(client, auth, "var_001_M", 6)
    error = assert_error(response, 409, "STOCK_INSUFFICIENT")
    assert error["details"]["available"] == 5

    cart = await get_cart(client, auth)
    assert cart.items == []
    assert cart.version == 0


async def test_out_of_stock_variant_cannot_be_added(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    error = assert_error(
        await add_item(client, auth, "var_001_L", 1), 409, "STOCK_INSUFFICIENT"
    )
    assert error["details"]["available"] == 0


async def test_unknown_variant_is_not_found(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    assert_error(await add_item(client, auth, "var_999_S", 1), 404, "NOT_FOUND")


async def test_cart_items_are_sorted_by_product_then_size(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    for variant in ("var_002_M", "var_001_XL", "var_001_S"):
        await add_item(client, auth, variant)
    cart = await get_cart(client, auth)
    assert [i.variant_id for i in cart.items] == [
        "var_001_S",
        "var_001_XL",
        "var_002_M",
    ]


async def test_unit_price_includes_variant_price_delta(
    client: AsyncClient, auth: dict[str, str], db_session: AsyncSession
) -> None:
    await db_session.execute(
        update(ProductVariant)
        .where(ProductVariant.public_id == "var_002_M")
        .values(price_delta=50000)
    )
    await db_session.commit()

    response = await add_item(client, auth, "var_002_M", 2)
    cart = c.CartSummary.model_validate(response.json())
    assert cart.items[0].unit_price == 570000
    assert cart.total_amount == 1140000


async def test_update_quantity(client: AsyncClient, auth: dict[str, str]) -> None:
    await add_item(client, auth, "var_001_M", 1)

    response = await client.patch(
        f"{BASE}/cart/items/var_001_M", json={"quantity": 3}, headers=auth
    )
    assert response.status_code == 200
    cart = c.CartSummary.model_validate(response.json())
    assert [i.quantity for i in cart.items] == [3]
    assert cart.version == 2

    over = await client.patch(
        f"{BASE}/cart/items/var_001_M", json={"quantity": 6}, headers=auth
    )
    assert_error(over, 409, "STOCK_INSUFFICIENT")

    missing = await client.patch(
        f"{BASE}/cart/items/var_002_M", json={"quantity": 1}, headers=auth
    )
    assert_error(missing, 404, "NOT_FOUND")


async def test_remove_item(client: AsyncClient, auth: dict[str, str]) -> None:
    await add_item(client, auth, "var_001_M", 1)

    response = await client.delete(f"{BASE}/cart/items/var_001_M", headers=auth)
    assert response.status_code == 200
    cart = c.CartSummary.model_validate(response.json())
    assert cart.items == []
    assert (cart.item_count, cart.total_amount, cart.version) == (0, 0, 2)

    again = await client.delete(f"{BASE}/cart/items/var_001_M", headers=auth)
    assert_error(again, 404, "NOT_FOUND")


async def test_if_match_detects_stale_version(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    assert (await get_cart(client, auth)).version == 0

    first = await add_item(client, auth, "var_001_S", 1, **{"If-Match": "0"})
    assert first.status_code == 200
    assert first.json()["version"] == 1

    stale = await add_item(client, auth, "var_001_S", 1, **{"If-Match": "0"})
    error = assert_error(stale, 409, "VERSION_CONFLICT")
    assert error["details"]["current_version"] == 1

    quoted = await add_item(client, auth, "var_001_S", 1, **{"If-Match": '"1"'})
    assert quoted.status_code == 200

    stale_patch = await client.patch(
        f"{BASE}/cart/items/var_001_S",
        json={"quantity": 1},
        headers={**auth, "If-Match": "1"},
    )
    assert_error(stale_patch, 409, "VERSION_CONFLICT")

    stale_delete = await client.delete(
        f"{BASE}/cart/items/var_001_S", headers={**auth, "If-Match": "1"}
    )
    assert_error(stale_delete, 409, "VERSION_CONFLICT")


async def test_if_match_must_be_a_number(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    response = await add_item(client, auth, "var_001_S", 1, **{"If-Match": "abc"})
    assert_error(response, 422, "VALIDATION_ERROR")


async def test_concurrent_writes_with_same_version_only_one_wins(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await get_cart(client, auth)  # creates the cart at version 0

    responses = await asyncio.gather(
        *[add_item(client, auth, "var_001_S", 1, **{"If-Match": "0"}) for _ in range(2)]
    )
    assert sorted(r.status_code for r in responses) == [200, 409]

    cart = await get_cart(client, auth)
    assert cart.version == 1
    assert cart.item_count == 1


async def test_carts_are_private_per_user(
    client: AsyncClient, auth: dict[str, str]
) -> None:
    await add_item(client, auth, "var_001_S", 1)
    other = await guest_headers(client)
    assert (await get_cart(client, other)).items == []
    assert (await get_cart(client, auth)).item_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"variant_id": "var_001_S", "quantity": 0},
        {"variant_id": "var_001_S", "quantity": -1},
        {"variant_id": "var_001_S", "quantity": "1"},
        {"variant_id": "var_001_S"},
        {"quantity": 1},
        {"variant_id": "not-a-variant", "quantity": 1},
        {"variant_id": "var_001_S", "quantity": 1, "extra": True},
    ],
)
async def test_add_item_validates_body(
    client: AsyncClient, auth: dict[str, str], payload: dict[str, Any]
) -> None:
    response = await client.post(f"{BASE}/cart/items", json=payload, headers=auth)
    error = assert_error(response, 422, "VALIDATION_ERROR")
    assert error["details"]["errors"]
