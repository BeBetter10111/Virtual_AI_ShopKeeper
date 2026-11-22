"""Pydantic v2 mirror of packages/shared-contracts (JSON Schema + TypeScript).

Models forbid extra fields and run in strict mode, matching the schemas'
`additionalProperties: false` and exact JSON types. Money fields are plain
integers: convert Decimal values from the ORM before building these models.
tests/test_contract_parity.py keeps this module and the schemas in sync.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

ProductId = Annotated[str, StringConstraints(pattern=r"^prod_[A-Za-z0-9_]+$")]
VariantId = Annotated[str, StringConstraints(pattern=r"^var_[A-Za-z0-9_]+$")]
OrderId = Annotated[str, StringConstraints(pattern=r"^ord_[A-Za-z0-9_]+$")]
WaypointId = Annotated[str, StringConstraints(pattern=r"^WP_[A-Z0-9_]+$")]

NonNegInt = Annotated[int, Field(ge=0)]
NonNegFloat = Annotated[float, Field(ge=0)]
Quantity = Annotated[int, Field(ge=1)]
Probability = Annotated[float, Field(ge=0, le=1)]
LabColor = Annotated[list[float], Field(min_length=3, max_length=3)]

Language = Literal["vi", "en"]
Viseme = Literal[
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O",
    "U",
]
AvatarAnimation = Literal[
    "IDLE",
    "WAVE",
    "NOD",
    "THINKING",
    "POINT_LEFT",
    "POINT_RIGHT",
    "GESTURE_GUIDE",
    "SHRUG",
]
GraphicsQuality = Literal["Low", "Medium", "High"]
QualityWarning = Literal["LOW_LIGHT", "OVEREXPOSED"]
ApiErrorCode = Literal[
    "UNAUTHORIZED",
    "VALIDATION_ERROR",
    "NOT_FOUND",
    "STOCK_INSUFFICIENT",
    "VERSION_CONFLICT",
    "IMAGE_INVALID",
    "RATE_LIMITED",
]
WsErrorCode = Literal["STT_FAILED", "CV_UNSUPPORTED_IMAGE", "RATE_LIMITED", "INTERNAL"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EmptyPayload(ContractModel):
    pass


class Vector3(ContractModel):
    x: float
    y: float
    z: float


class VisemeFrame(ContractModel):
    time_ms: NonNegInt
    viseme: Viseme


# ---------------------------------------------------------------------------
# REST DTOs (/api/v1)
# ---------------------------------------------------------------------------


class ApiErrorBody(ContractModel):
    code: ApiErrorCode
    message: str
    details: dict[str, Any] | None = None


class ApiError(ContractModel):
    error: ApiErrorBody


class HealthResponse(ContractModel):
    status: str
    postgres: str
    redis: str


class SessionRequest(ContractModel):
    mode: Literal["guest", "login"]
    email: str | None = None
    password: Annotated[str, Field(min_length=8)] | None = None
    accessibility_mode: bool = False

    @model_validator(mode="after")
    def login_requires_credentials(self) -> Self:
        if self.mode == "login" and (self.email is None or self.password is None):
            raise ValueError("email and password are required when mode is 'login'")
        return self


class RegisterRequest(ContractModel):
    email: str
    password: Annotated[str, Field(min_length=8)]
    accessibility_mode: bool = False


class SessionResponse(ContractModel):
    access_token: str
    session_id: str
    user_id: str
    expires_in: Annotated[int, Field(ge=1)]


class ShelfProduct(ContractModel):
    product_id: ProductId
    unity_prefab_id: str
    glb_model_url: str | None = None
    name: str
    base_price: NonNegInt


class Shelf(ContractModel):
    waypoint_id: WaypointId
    aisle_code: str
    position: Vector3
    products: list[ShelfProduct]


class StoreLayout(ContractModel):
    shelves: list[Shelf]
    checkout_waypoint_id: WaypointId


class ProductVariant(ContractModel):
    variant_id: VariantId
    size: str
    color_name: str
    stock_qty: NonNegInt
    price_delta: int


class ProductDetail(ContractModel):
    product_id: ProductId
    name: str
    description: str
    base_price: NonNegInt
    glb_model_url: str | None = None
    waypoint_id: WaypointId
    variants: list[ProductVariant]


class VisualSearchMatch(ContractModel):
    product_id: ProductId
    name: str
    similarity_score: Probability
    delta_e00: NonNegFloat
    waypoint_id: WaypointId
    in_stock: bool


class VisualSearchResponse(ContractModel):
    detected_color_name: str
    detected_lab: LabColor
    category_label: str
    category_confidence: Probability
    is_in_domain: bool
    quality_warning: QualityWarning | None
    matches: Annotated[list[VisualSearchMatch], Field(max_length=5)]
    alternatives: Annotated[list[VisualSearchMatch], Field(max_length=2)]


class AddCartItemRequest(ContractModel):
    variant_id: VariantId
    quantity: Quantity


class UpdateCartItemRequest(ContractModel):
    quantity: Quantity


class CartItem(ContractModel):
    product_id: ProductId
    variant_id: VariantId
    name: str
    size: str
    quantity: Quantity
    unit_price: NonNegInt


class CartSummary(ContractModel):
    cart_id: str
    version: NonNegInt
    items: list[CartItem]
    item_count: NonNegInt
    total_amount: NonNegInt


class ShippingInfo(ContractModel):
    full_name: Annotated[str, Field(min_length=1)]
    address: Annotated[str, Field(min_length=1)]
    phone: Annotated[str, Field(min_length=1)]


class CheckoutRequest(ContractModel):
    shipping_info: ShippingInfo
    payment_method: Literal["SIMULATED"]


class Order(ContractModel):
    order_id: OrderId
    status: Literal["CONFIRMED"]
    total_amount: NonNegInt
    items: Annotated[list[CartItem], Field(min_length=1)]


# ---------------------------------------------------------------------------
# Unity bridge: React -> Unity commands
# ---------------------------------------------------------------------------


class InitStoreLayoutCommand(ContractModel):
    action: Literal["INIT_STORE_LAYOUT"]
    payload: StoreLayout


class NavigateToWaypointPayload(ContractModel):
    waypoint_id: WaypointId
    stop_distance_m: NonNegFloat = 1.5


class NavigateToWaypointCommand(ContractModel):
    action: Literal["NAVIGATE_TO_WAYPOINT"]
    payload: NavigateToWaypointPayload


class CancelNavigationCommand(ContractModel):
    action: Literal["CANCEL_NAVIGATION"]
    payload: EmptyPayload


class HighlightProductsPayload(ContractModel):
    product_ids: list[ProductId]
    duration_ms: NonNegInt = 8000


class HighlightProductsCommand(ContractModel):
    action: Literal["HIGHLIGHT_PRODUCTS"]
    payload: HighlightProductsPayload


class PlayAvatarAnimationPayload(ContractModel):
    animation: AvatarAnimation


class PlayAvatarAnimationCommand(ContractModel):
    action: Literal["PLAY_AVATAR_ANIMATION"]
    payload: PlayAvatarAnimationPayload


class FeedVisemeDataPayload(ContractModel):
    response_id: str
    chunk_seq: NonNegInt
    playback_latency_ms: NonNegFloat
    visemes: list[VisemeFrame]


class FeedVisemeDataCommand(ContractModel):
    action: Literal["FEED_VISEME_DATA"]
    payload: FeedVisemeDataPayload


class StopSpeakingPayload(ContractModel):
    response_id: str


class StopSpeakingCommand(ContractModel):
    action: Literal["STOP_SPEAKING"]
    payload: StopSpeakingPayload


class SetGraphicsQualityPayload(ContractModel):
    level: GraphicsQuality


class SetGraphicsQualityCommand(ContractModel):
    action: Literal["SET_GRAPHICS_QUALITY"]
    payload: SetGraphicsQualityPayload


class UpdateCartBadgePayload(ContractModel):
    item_count: NonNegInt
    total_amount: NonNegInt


class UpdateCartBadgeCommand(ContractModel):
    action: Literal["UPDATE_CART_BADGE"]
    payload: UpdateCartBadgePayload


UnityCommand = Annotated[
    InitStoreLayoutCommand
    | NavigateToWaypointCommand
    | CancelNavigationCommand
    | HighlightProductsCommand
    | PlayAvatarAnimationCommand
    | FeedVisemeDataCommand
    | StopSpeakingCommand
    | SetGraphicsQualityCommand
    | UpdateCartBadgeCommand,
    Field(discriminator="action"),
]


# ---------------------------------------------------------------------------
# Unity bridge: Unity -> React events
# ---------------------------------------------------------------------------


class PlayerMovedPayload(ContractModel):
    position: Vector3
    yaw: float
    pitch: float
    current_aisle: str | None
    gaze_target_product_id: ProductId | None


class StoreReadyEvent(ContractModel):
    event: Literal["ON_STORE_READY"]
    payload: EmptyPayload


class PlayerMovedEvent(ContractModel):
    event: Literal["ON_PLAYER_MOVED"]
    payload: PlayerMovedPayload


class ProductInspectClickedPayload(ContractModel):
    product_id: ProductId


class ProductInspectClickedEvent(ContractModel):
    event: Literal["ON_PRODUCT_INSPECT_CLICKED"]
    payload: ProductInspectClickedPayload


class WaypointArrivedPayload(ContractModel):
    waypoint_id: WaypointId


class WaypointArrivedEvent(ContractModel):
    event: Literal["ON_WAYPOINT_ARRIVED"]
    payload: WaypointArrivedPayload


class NavigationFailedPayload(ContractModel):
    waypoint_id: str
    reason: Literal["UNREACHABLE", "CANCELLED", "UNKNOWN_WAYPOINT"]


class NavigationFailedEvent(ContractModel):
    event: Literal["ON_NAVIGATION_FAILED"]
    payload: NavigationFailedPayload


class FpsDropPayload(ContractModel):
    fps: NonNegFloat
    duration_ms: NonNegInt
    applied_quality: GraphicsQuality


class FpsDropDetectedEvent(ContractModel):
    event: Literal["ON_FPS_DROP_DETECTED"]
    payload: FpsDropPayload


class CheckoutCounterEnteredEvent(ContractModel):
    event: Literal["ON_CHECKOUT_COUNTER_ENTERED"]
    payload: EmptyPayload


UnityEvent = Annotated[
    StoreReadyEvent
    | PlayerMovedEvent
    | ProductInspectClickedEvent
    | WaypointArrivedEvent
    | NavigationFailedEvent
    | FpsDropDetectedEvent
    | CheckoutCounterEnteredEvent,
    Field(discriminator="event"),
]


# ---------------------------------------------------------------------------
# WebSocket: client -> server
# ---------------------------------------------------------------------------


class AudioEndPayload(ContractModel):
    audio_mode: Literal["ptt", "vad"]


class AudioEndMessage(ContractModel):
    type: Literal["AUDIO_END"]
    timestamp: int
    payload: AudioEndPayload


class TextMessagePayload(ContractModel):
    text: Annotated[str, Field(min_length=1)]
    language: Language
    want_audio: bool = False


class TextMessage(ContractModel):
    type: Literal["TEXT_MESSAGE"]
    timestamp: int
    payload: TextMessagePayload


class CvFrameRoiPayload(ContractModel):
    image_base64: str
    hand_confidence: Probability
    gaze_target_product_id: ProductId | None


class CvFrameRoiMessage(ContractModel):
    type: Literal["CV_FRAME_ROI"]
    timestamp: int
    payload: CvFrameRoiPayload


class PlayerTelemetryMessage(ContractModel):
    type: Literal["PLAYER_TELEMETRY"]
    timestamp: int
    payload: PlayerMovedPayload


class BargeInInterruptPayload(ContractModel):
    response_id: str


class BargeInInterruptMessage(ContractModel):
    type: Literal["BARGE_IN_INTERRUPT"]
    timestamp: int
    payload: BargeInInterruptPayload


class HeartbeatMessage(ContractModel):
    """Sent by both sides."""

    type: Literal["HEARTBEAT"]
    timestamp: int
    payload: EmptyPayload


class ResyncRequestMessage(ContractModel):
    type: Literal["RESYNC_REQUEST"]
    timestamp: int
    payload: EmptyPayload


ClientMessage = Annotated[
    AudioEndMessage
    | TextMessage
    | CvFrameRoiMessage
    | PlayerTelemetryMessage
    | BargeInInterruptMessage
    | HeartbeatMessage
    | ResyncRequestMessage,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# WebSocket: server -> client
# ---------------------------------------------------------------------------


class SpatialState(ContractModel):
    position: Vector3
    current_aisle: str | None


class SessionSyncPayload(ContractModel):
    cart: CartSummary
    spatial: SpatialState
    last_waypoint_id: str | None


class SessionSyncMessage(ContractModel):
    type: Literal["SESSION_SYNC"]
    timestamp: int
    payload: SessionSyncPayload


class SttTranscriptPayload(ContractModel):
    text: str
    confidence: Probability
    is_final: bool
    language: Language


class SttTranscriptMessage(ContractModel):
    type: Literal["STT_TRANSCRIPT"]
    timestamp: int
    payload: SttTranscriptPayload


class AgentResponseChunk(ContractModel):
    response_id: str
    chunk_seq: NonNegInt
    is_final: bool
    text: str
    audio_base64: str | None
    audio_format: Literal["mp3", "opus"] | None
    audio_duration_ms: NonNegInt
    # time_ms is relative to the start of this chunk; empty for text-only replies.
    visemes: list[VisemeFrame]
    recommended_products: list[ProductId] | None = None
    avatar_animation: AvatarAnimation | None = None


class AgentResponseStreamMessage(ContractModel):
    type: Literal["AGENT_RESPONSE_STREAM"]
    timestamp: int
    payload: AgentResponseChunk


class ResponseCancelledPayload(ContractModel):
    response_id: str


class ResponseCancelledMessage(ContractModel):
    type: Literal["RESPONSE_CANCELLED"]
    timestamp: int
    payload: ResponseCancelledPayload


class UnityCommandPayload(ContractModel):
    response_id: str
    commands: list[UnityCommand]


class UnityCommandMessage(ContractModel):
    type: Literal["UNITY_COMMAND"]
    timestamp: int
    payload: UnityCommandPayload


class OpenProductInspectorAction(ContractModel):
    action: Literal["OPEN_PRODUCT_INSPECTOR"]
    product_id: ProductId


class OpenCheckoutAction(ContractModel):
    action: Literal["OPEN_CHECKOUT"]


class CloseModalAction(ContractModel):
    action: Literal["CLOSE_MODAL"]


UiActionPayload = Annotated[
    OpenProductInspectorAction | OpenCheckoutAction | CloseModalAction,
    Field(discriminator="action"),
]


class UiActionMessage(ContractModel):
    type: Literal["UI_ACTION"]
    timestamp: int
    payload: UiActionPayload


class CartUpdatedMessage(ContractModel):
    type: Literal["CART_UPDATED"]
    timestamp: int
    payload: CartSummary


class CvQualityWarningPayload(ContractModel):
    warning: QualityWarning
    mean_luminance: Annotated[float, Field(ge=0, le=255)]


class CvQualityWarningMessage(ContractModel):
    type: Literal["CV_QUALITY_WARNING"]
    timestamp: int
    payload: CvQualityWarningPayload


class WsErrorPayload(ContractModel):
    code: WsErrorCode
    message: str
    recoverable: bool


class WsErrorMessage(ContractModel):
    type: Literal["ERROR"]
    timestamp: int
    payload: WsErrorPayload


ServerMessage = Annotated[
    SessionSyncMessage
    | SttTranscriptMessage
    | AgentResponseStreamMessage
    | ResponseCancelledMessage
    | UnityCommandMessage
    | UiActionMessage
    | CartUpdatedMessage
    | CvQualityWarningMessage
    | HeartbeatMessage
    | WsErrorMessage,
    Field(discriminator="type"),
]
