"""Keeps packages/shared-contracts (JSON Schema) and app.schemas.contracts in sync.

Three checks:
1. The same payloads are accepted/rejected by both sides.
2. Enum values are identical.
3. Message/command/event sets and top-level field/required lists are identical.
"""

import json
import typing
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, TypeAdapter, ValidationError
from referencing import Registry, Resource

from app.schemas import contracts as c

SCHEMA_DIR = (
    Path(__file__).resolve().parents[3] / "packages" / "shared-contracts" / "schemas"
)
SCHEMAS: dict[str, dict[str, Any]] = {
    name: json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    for name in ("inventory", "unity_bridge", "ws_events")
}
REGISTRY: Registry = Registry().with_resources(
    [(s["$id"], Resource.from_contents(s)) for s in SCHEMAS.values()]
)


def schema_accepts(file: str, definition: str, payload: Any) -> bool:
    ref = {"$ref": f"{SCHEMAS[file]['$id']}#/$defs/{definition}"}
    return Draft202012Validator(ref, registry=REGISTRY).is_valid(payload)


def pydantic_accepts(model: Any, payload: Any) -> bool:
    try:
        TypeAdapter(model).validate_python(payload)
    except ValidationError:
        return False
    return True


# ---------------------------------------------------------------------------
# 1. Payload parity
# ---------------------------------------------------------------------------

NOW = 1795015500123
CART_ITEM = {
    "product_id": "prod_001",
    "variant_id": "var_001_L",
    "name": "Blue Oxford Shirt",
    "size": "L",
    "quantity": 1,
    "unit_price": 450000,
}
CART = {
    "cart_id": "c1",
    "version": 7,
    "items": [CART_ITEM],
    "item_count": 1,
    "total_amount": 450000,
}
MATCH = {
    "product_id": "prod_001",
    "name": "Blue Oxford Shirt",
    "similarity_score": 0.92,
    "delta_e00": 3.4,
    "waypoint_id": "WP_AISLE_01_SHIRTS",
    "in_stock": True,
}
VISUAL_SEARCH = {
    "detected_color_name": "Navy Blue",
    "detected_lab": [28.4, 6.2, -31.5],
    "category_label": "shirt",
    "category_confidence": 0.88,
    "is_in_domain": True,
    "quality_warning": None,
    "matches": [MATCH],
    "alternatives": [],
}
SHELF = {
    "waypoint_id": "WP_AISLE_01_SHIRTS",
    "aisle_code": "A1",
    "position": {"x": 12.5, "y": 0, "z": -4.2},
    "products": [
        {
            "product_id": "prod_001",
            "unity_prefab_id": "Pref_BlueOxfordShirt",
            "glb_model_url": "https://example.test/prod_001.glb",
            "name": "Blue Oxford Shirt",
            "base_price": 450000,
        }
    ],
}
PLAYER = {
    "position": {"x": 4.2, "y": 0, "z": 1.8},
    "yaw": 90,
    "pitch": -5,
    "current_aisle": "A1",
    "gaze_target_product_id": "prod_001",
}
CHUNK = {
    "response_id": "resp_0042",
    "chunk_seq": 0,
    "is_final": False,
    "text": "Here are some shirts.",
    "audio_base64": "AAAA",
    "audio_format": "mp3",
    "audio_duration_ms": 3200,
    "visemes": [{"time_ms": 0, "viseme": "sil"}, {"time_ms": 120, "viseme": "aa"}],
    "recommended_products": ["prod_001"],
    "avatar_animation": "NOD",
}


def msg(type_: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": type_, "timestamp": NOW, "payload": payload}


def cmd(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"action": action, "payload": payload}


# (schema file, definition, python type, payload, should be valid)
CASES: list[tuple[str, str, Any, Any, bool]] = [
    ("inventory", "CartSummary", c.CartSummary, CART, True),
    ("inventory", "CartSummary", c.CartSummary, {**CART, "version": "7"}, False),
    ("inventory", "CartSummary", c.CartSummary, {**CART, "extra": 1}, False),
    ("inventory", "SessionRequest", c.SessionRequest, {"mode": "guest"}, True),
    (
        "inventory",
        "SessionRequest",
        c.SessionRequest,
        {"mode": "login", "email": "a@b.co", "password": "longenough"},
        True,
    ),
    ("inventory", "SessionRequest", c.SessionRequest, {"mode": "login"}, False),
    (
        "inventory",
        "SessionRequest",
        c.SessionRequest,
        {"mode": "login", "email": "a@b.co", "password": "short"},
        False,
    ),
    (
        "inventory",
        "StoreLayout",
        c.StoreLayout,
        {"shelves": [SHELF], "checkout_waypoint_id": "WP_CHECKOUT"},
        True,
    ),
    (
        "inventory",
        "ShelfProduct",
        c.ShelfProduct,
        {**SHELF["products"][0], "product_id": "001"},
        False,
    ),
    (
        "inventory",
        "ProductDetail",
        c.ProductDetail,
        {
            "product_id": "prod_001",
            "name": "Blue Oxford Shirt",
            "description": "d",
            "base_price": 450000,
            "waypoint_id": "WP_AISLE_01_SHIRTS",
            "variants": [
                {
                    "variant_id": "var_001_L",
                    "size": "L",
                    "color_name": "Navy Blue",
                    "stock_qty": 0,
                    "price_delta": 0,
                }
            ],
        },
        True,
    ),
    ("inventory", "VisualSearchResponse", c.VisualSearchResponse, VISUAL_SEARCH, True),
    (
        "inventory",
        "VisualSearchResponse",
        c.VisualSearchResponse,
        {**VISUAL_SEARCH, "quality_warning": "LOW_LIGHT"},
        True,
    ),
    (
        "inventory",
        "VisualSearchResponse",
        c.VisualSearchResponse,
        {**VISUAL_SEARCH, "quality_warning": "DARK"},
        False,
    ),
    (
        "inventory",
        "VisualSearchResponse",
        c.VisualSearchResponse,
        {**VISUAL_SEARCH, "matches": [MATCH] * 6},
        False,
    ),
    (
        "inventory",
        "VisualSearchResponse",
        c.VisualSearchResponse,
        {**VISUAL_SEARCH, "detected_lab": [1.0, 2.0]},
        False,
    ),
    (
        "inventory",
        "ApiError",
        c.ApiError,
        {"error": {"code": "STOCK_INSUFFICIENT", "message": "m", "details": {"n": 2}}},
        True,
    ),
    (
        "inventory",
        "ApiError",
        c.ApiError,
        {"error": {"code": "TEAPOT", "message": "m"}},
        False,
    ),
    (
        "inventory",
        "CheckoutRequest",
        c.CheckoutRequest,
        {
            "shipping_info": {"full_name": "A", "address": "B", "phone": "C"},
            "payment_method": "SIMULATED",
        },
        True,
    ),
    (
        "inventory",
        "CheckoutRequest",
        c.CheckoutRequest,
        {
            "shipping_info": {"full_name": "A", "address": "B", "phone": "C"},
            "payment_method": "CARD",
        },
        False,
    ),
    (
        "inventory",
        "Order",
        c.Order,
        {
            "order_id": "ord_0001",
            "status": "CONFIRMED",
            "total_amount": 450000,
            "items": [CART_ITEM],
        },
        True,
    ),
    (
        "inventory",
        "Order",
        c.Order,
        {
            "order_id": "ord_0001",
            "status": "CONFIRMED",
            "total_amount": 450000,
            "items": [],
        },
        False,
    ),
    # Unity commands
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd("NAVIGATE_TO_WAYPOINT", {"waypoint_id": "WP_AISLE_01_SHIRTS"}),
        True,
    ),
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd("NAVIGATE_TO_WAYPOINT", {"navigate_waypoint_id": "WP_X"}),
        False,
    ),
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd("HIGHLIGHT_PRODUCTS", {"product_ids": [], "duration_ms": 8000}),
        True,
    ),
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd("CANCEL_NAVIGATION", {}),
        True,
    ),
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd(
            "FEED_VISEME_DATA",
            {
                "response_id": "r",
                "chunk_seq": 0,
                "playback_latency_ms": 20,
                "visemes": [{"time_ms": 0, "viseme": "sil"}],
            },
        ),
        True,
    ),
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd(
            "FEED_VISEME_DATA",
            {
                "response_id": "r",
                "chunk_seq": 0,
                "playback_latency_ms": 20,
                "visemes": [{"time_ms": 0, "viseme": "zz"}],
            },
        ),
        False,
    ),
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd("TELEPORT", {}),
        False,
    ),
    (
        "unity_bridge",
        "UnityCommand",
        c.UnityCommand,
        cmd("PLAY_AVATAR_ANIMATION", {"animation": "DANCE"}),
        False,
    ),
    # Unity events
    (
        "unity_bridge",
        "UnityEvent",
        c.UnityEvent,
        {"event": "ON_PLAYER_MOVED", "payload": PLAYER},
        True,
    ),
    (
        "unity_bridge",
        "UnityEvent",
        c.UnityEvent,
        {
            "event": "ON_NAVIGATION_FAILED",
            "payload": {"waypoint_id": "WP_X", "reason": "UNREACHABLE"},
        },
        True,
    ),
    (
        "unity_bridge",
        "UnityEvent",
        c.UnityEvent,
        {
            "event": "ON_NAVIGATION_FAILED",
            "payload": {"waypoint_id": "WP_X", "reason": "BORED"},
        },
        False,
    ),
    # WebSocket: server -> client
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("AGENT_RESPONSE_STREAM", CHUNK),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg(
            "AGENT_RESPONSE_STREAM",
            {**CHUNK, "audio_base64": None, "audio_format": None, "visemes": []},
        ),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("AGENT_RESPONSE_STREAM", {**CHUNK, "viseme_timeline": []}),
        False,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        {"type": "AGENT_RESPONSE_STREAM", "payload": CHUNK},
        False,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg(
            "UNITY_COMMAND",
            {
                "response_id": "resp_0042",
                "commands": [
                    cmd("PLAY_AVATAR_ANIMATION", {"animation": "GESTURE_GUIDE"}),
                    cmd("NAVIGATE_TO_WAYPOINT", {"waypoint_id": "WP_AISLE_01_SHIRTS"}),
                    cmd(
                        "HIGHLIGHT_PRODUCTS",
                        {"product_ids": ["prod_001"], "duration_ms": 8000},
                    ),
                ],
            },
        ),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("UNITY_COMMAND", {"response_id": "r", "commands": [cmd("TELEPORT", {})]}),
        False,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg(
            "UI_ACTION",
            {"action": "OPEN_PRODUCT_INSPECTOR", "product_id": "prod_001"},
        ),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("UI_ACTION", {"action": "OPEN_CHECKOUT"}),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("UI_ACTION", {"action": "OPEN_PRODUCT_INSPECTOR"}),
        False,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg(
            "SESSION_SYNC",
            {
                "cart": CART,
                "spatial": {
                    "position": {"x": 4.2, "y": 0, "z": 1.8},
                    "current_aisle": "A1",
                },
                "last_waypoint_id": "WP_AISLE_01_SHIRTS",
            },
        ),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("CART_UPDATED", CART),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg(
            "STT_TRANSCRIPT",
            {"text": "hi", "confidence": 0.93, "is_final": True, "language": "vi"},
        ),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg(
            "STT_TRANSCRIPT",
            {"text": "hi", "confidence": 1.5, "is_final": True, "language": "vi"},
        ),
        False,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("CV_QUALITY_WARNING", {"warning": "LOW_LIGHT", "mean_luminance": 31}),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("ERROR", {"code": "STT_FAILED", "message": "m", "recoverable": True}),
        True,
    ),
    (
        "ws_events",
        "ServerMessage",
        c.ServerMessage,
        msg("RESPONSE_CANCELLED", {"response_id": "resp_0042"}),
        True,
    ),
    ("ws_events", "ServerMessage", c.ServerMessage, msg("HEARTBEAT", {}), True),
    # WebSocket: client -> server
    ("ws_events", "ClientMessage", c.ClientMessage, msg("HEARTBEAT", {}), True),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg("AUDIO_END", {"audio_mode": "ptt"}),
        True,
    ),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg("TEXT_MESSAGE", {"text": "Take me to jackets", "language": "en"}),
        True,
    ),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg("TEXT_MESSAGE", {"text": "", "language": "en"}),
        False,
    ),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg(
            "CV_FRAME_ROI",
            {
                "image_base64": "/9j/",
                "hand_confidence": 0.94,
                "gaze_target_product_id": None,
            },
        ),
        True,
    ),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg(
            "CV_FRAME_ROI",
            {
                "image_base64": "/9j/",
                "hand_confidence": 1.5,
                "gaze_target_product_id": None,
            },
        ),
        False,
    ),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg("PLAYER_TELEMETRY", PLAYER),
        True,
    ),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg("BARGE_IN_INTERRUPT", {"response_id": "resp_0042"}),
        True,
    ),
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg("BARGE_IN_INTERRUPT", {}),
        False,
    ),
    ("ws_events", "ClientMessage", c.ClientMessage, msg("RESYNC_REQUEST", {}), True),
    # A server-only message type must not be accepted from a client.
    (
        "ws_events",
        "ClientMessage",
        c.ClientMessage,
        msg("RESPONSE_CANCELLED", {"response_id": "r"}),
        False,
    ),
]


@pytest.mark.parametrize(
    ("file", "definition", "model", "payload", "valid"),
    CASES,
    ids=[
        f"{i:02d}-{case[1]}-{'ok' if case[4] else 'bad'}"
        for i, case in enumerate(CASES)
    ],
)
def test_payload_parity(
    file: str, definition: str, model: Any, payload: Any, valid: bool
) -> None:
    assert schema_accepts(file, definition, payload) is valid, "JSON Schema disagrees"
    assert pydantic_accepts(model, payload) is valid, "Pydantic disagrees"


# ---------------------------------------------------------------------------
# 2. Enum parity
# ---------------------------------------------------------------------------

ENUMS: list[tuple[str, str, Any]] = [
    ("unity_bridge", "Viseme", c.Viseme),
    ("unity_bridge", "AvatarAnimation", c.AvatarAnimation),
    ("unity_bridge", "GraphicsQuality", c.GraphicsQuality),
    ("inventory", "QualityWarning", c.QualityWarning),
    ("inventory", "ApiErrorCode", c.ApiErrorCode),
    ("ws_events", "Language", c.Language),
]


@pytest.mark.parametrize(("file", "definition", "literal"), ENUMS)
def test_enum_parity(file: str, definition: str, literal: Any) -> None:
    schema_values = SCHEMAS[file]["$defs"][definition]["enum"]
    assert set(schema_values) == set(typing.get_args(literal))


def test_ws_error_codes_match() -> None:
    schema_values = SCHEMAS["ws_events"]["$defs"]["ERROR"]["properties"]["payload"][
        "properties"
    ]["code"]["enum"]
    assert set(schema_values) == set(typing.get_args(c.WsErrorCode))


# ---------------------------------------------------------------------------
# 3. Structure parity
# ---------------------------------------------------------------------------


def _union_members(annotated_union: Any) -> list[type[BaseModel]]:
    union = typing.get_args(annotated_union)[0]
    return list(typing.get_args(union))


def _literal_value(model: type[BaseModel], field: str) -> str:
    (value,) = typing.get_args(model.model_fields[field].annotation)
    return str(value)


def _required(model: type[BaseModel]) -> set[str]:
    return {name for name, info in model.model_fields.items() if info.is_required()}


UNIONS: list[tuple[str, str, Any, str]] = [
    ("ws_events", "ClientMessage", c.ClientMessage, "type"),
    ("ws_events", "ServerMessage", c.ServerMessage, "type"),
    ("unity_bridge", "UnityCommand", c.UnityCommand, "action"),
    ("unity_bridge", "UnityEvent", c.UnityEvent, "event"),
]


@pytest.mark.parametrize(("file", "union_name", "union", "discriminator"), UNIONS)
def test_union_members_match(
    file: str, union_name: str, union: Any, discriminator: str
) -> None:
    defs = SCHEMAS[file]["$defs"]
    schema_members: dict[str, dict[str, Any]] = {}
    for option in defs[union_name]["oneOf"]:
        member = defs[option["$ref"].rsplit("/", 1)[1]]
        const = member["properties"][discriminator]["const"]
        schema_members[const] = member

    python_members = {
        _literal_value(m, discriminator): m for m in _union_members(union)
    }

    assert set(schema_members) == set(python_members)
    for const, member in schema_members.items():
        model = python_members[const]
        assert set(member["properties"]) == set(model.model_fields), const
        assert set(member["required"]) == _required(model), const


INVENTORY_MODELS = [
    "ApiError",
    "HealthResponse",
    "SessionRequest",
    "RegisterRequest",
    "SessionResponse",
    "ShelfProduct",
    "Shelf",
    "StoreLayout",
    "ProductVariant",
    "ProductDetail",
    "VisualSearchMatch",
    "VisualSearchResponse",
    "AddCartItemRequest",
    "UpdateCartItemRequest",
    "CartItem",
    "CartSummary",
    "ShippingInfo",
    "CheckoutRequest",
    "Order",
    "Vector3",
]


@pytest.mark.parametrize("name", INVENTORY_MODELS)
def test_inventory_model_fields_match(name: str) -> None:
    definition = SCHEMAS["inventory"]["$defs"][name]
    model: type[BaseModel] = getattr(c, name)
    assert set(definition["properties"]) == set(model.model_fields)
    assert set(definition["required"]) == _required(model)
