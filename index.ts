/**
 * Shared contract types. Hand-written mirror of ../schemas/*.json.
 * Field names follow the canonical naming table in ARCHITECTURE.md section 1.5.
 * Do not add `any`; keep in sync with schemas (parity is tested in apps/api).
 */

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/** Public string IDs; internal DB UUIDs never cross the API boundary. */
export type ProductId = string; // e.g. "prod_001"
export type VariantId = string; // e.g. "var_001_L"
export type OrderId = string; // e.g. "ord_0001"
export type WaypointId = string; // e.g. "WP_AISLE_01_SHIRTS"

export type Language = "vi" | "en";

export interface Vector3 {
  x: number;
  y: number;
  z: number;
}

/** CIELAB [L, a, b]. */
export type LabColor = [number, number, number];

/** Oculus 15 viseme set. */
export const VISEMES = [
  "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U",
] as const;
export type Viseme = (typeof VISEMES)[number];

export const AVATAR_ANIMATIONS = [
  "IDLE", "WAVE", "NOD", "THINKING", "POINT_LEFT", "POINT_RIGHT", "GESTURE_GUIDE", "SHRUG",
] as const;
export type AvatarAnimation = (typeof AVATAR_ANIMATIONS)[number];

export interface VisemeFrame {
  /** Relative to the start of the audio chunk. */
  time_ms: number;
  viseme: Viseme;
}

export type GraphicsQuality = "Low" | "Medium" | "High";
export type QualityWarning = "LOW_LIGHT" | "OVEREXPOSED";

// ---------------------------------------------------------------------------
// REST DTOs (/api/v1)
// ---------------------------------------------------------------------------

export type ApiErrorCode =
  | "UNAUTHORIZED"
  | "VALIDATION_ERROR"
  | "NOT_FOUND"
  | "STOCK_INSUFFICIENT"
  | "VERSION_CONFLICT"
  | "IMAGE_INVALID"
  | "RATE_LIMITED";

export interface ApiError {
  error: {
    code: ApiErrorCode;
    message: string;
    details?: Record<string, unknown>;
  };
}

export interface HealthResponse {
  status: string;
  postgres: string;
  redis: string;
}

export type SessionRequest =
  | { mode: "guest"; accessibility_mode?: boolean }
  | { mode: "login"; email: string; password: string; accessibility_mode?: boolean };

export interface RegisterRequest {
  email: string;
  password: string;
  accessibility_mode?: boolean;
}

export interface SessionResponse {
  access_token: string;
  session_id: string;
  user_id: string;
  expires_in: number;
}

export interface ShelfProduct {
  product_id: ProductId;
  unity_prefab_id: string;
  glb_model_url?: string | null;
  name: string;
  base_price: number;
}

export interface Shelf {
  waypoint_id: WaypointId;
  aisle_code: string;
  position: Vector3;
  products: ShelfProduct[];
}

export interface StoreLayout {
  shelves: Shelf[];
  checkout_waypoint_id: WaypointId;
}

export interface ProductVariant {
  variant_id: VariantId;
  size: string;
  color_name: string;
  stock_qty: number;
  price_delta: number;
}

export interface ProductDetail {
  product_id: ProductId;
  name: string;
  description: string;
  base_price: number;
  glb_model_url?: string | null;
  waypoint_id: WaypointId;
  variants: ProductVariant[];
}

export interface VisualSearchMatch {
  product_id: ProductId;
  name: string;
  similarity_score: number;
  delta_e00: number;
  waypoint_id: WaypointId;
  in_stock: boolean;
}

export interface VisualSearchResponse {
  detected_color_name: string;
  detected_lab: LabColor;
  category_label: string;
  category_confidence: number;
  /** true when category_confidence >= 0.45 */
  is_in_domain: boolean;
  quality_warning: QualityWarning | null;
  /** At most 5 entries. */
  matches: VisualSearchMatch[];
  /** At most 2 in-stock entries; only filled when the best match is out of stock. */
  alternatives: VisualSearchMatch[];
}

export interface AddCartItemRequest {
  variant_id: VariantId;
  quantity: number;
}

export interface UpdateCartItemRequest {
  quantity: number;
}

export interface CartItem {
  product_id: ProductId;
  variant_id: VariantId;
  name: string;
  size: string;
  quantity: number;
  unit_price: number;
}

export interface CartSummary {
  cart_id: string;
  /** Optimistic-lock version; sent back as the If-Match header. */
  version: number;
  items: CartItem[];
  item_count: number;
  total_amount: number;
}

export interface ShippingInfo {
  full_name: string;
  address: string;
  phone: string;
}

export interface CheckoutRequest {
  shipping_info: ShippingInfo;
  payment_method: "SIMULATED";
}

export interface Order {
  order_id: OrderId;
  status: "CONFIRMED";
  total_amount: number;
  items: CartItem[];
}

// ---------------------------------------------------------------------------
// Unity bridge
// ---------------------------------------------------------------------------

/** Payload of ON_PLAYER_MOVED (Unity -> React) and PLAYER_TELEMETRY (React -> server). */
export interface PlayerMovedPayload {
  position: Vector3;
  yaw: number;
  pitch: number;
  current_aisle: string | null;
  gaze_target_product_id: ProductId | null;
}

/** React -> Unity, via SendMessage("WebGLBridgeController", "DispatchCommand", json). */
export type UnityCommand =
  | { action: "INIT_STORE_LAYOUT"; payload: StoreLayout }
  | { action: "NAVIGATE_TO_WAYPOINT"; payload: { waypoint_id: WaypointId; stop_distance_m?: number } }
  | { action: "CANCEL_NAVIGATION"; payload: Record<string, never> }
  | { action: "HIGHLIGHT_PRODUCTS"; payload: { product_ids: ProductId[]; duration_ms?: number } }
  | { action: "PLAY_AVATAR_ANIMATION"; payload: { animation: AvatarAnimation } }
  | {
      action: "FEED_VISEME_DATA";
      payload: {
        response_id: string;
        chunk_seq: number;
        playback_latency_ms: number;
        visemes: VisemeFrame[];
      };
    }
  | { action: "STOP_SPEAKING"; payload: { response_id: string } }
  | { action: "SET_GRAPHICS_QUALITY"; payload: { level: GraphicsQuality } }
  | { action: "UPDATE_CART_BADGE"; payload: { item_count: number; total_amount: number } };

export type UnityCommandAction = UnityCommand["action"];

/** Unity -> React, delivered as window.onUnityEvent(event, JSON.stringify(payload)). */
export type UnityEvent =
  | { event: "ON_STORE_READY"; payload: Record<string, never> }
  | { event: "ON_PLAYER_MOVED"; payload: PlayerMovedPayload }
  | { event: "ON_PRODUCT_INSPECT_CLICKED"; payload: { product_id: ProductId } }
  | { event: "ON_WAYPOINT_ARRIVED"; payload: { waypoint_id: WaypointId } }
  | {
      event: "ON_NAVIGATION_FAILED";
      payload: { waypoint_id: string; reason: "UNREACHABLE" | "CANCELLED" | "UNKNOWN_WAYPOINT" };
    }
  | {
      event: "ON_FPS_DROP_DETECTED";
      payload: { fps: number; duration_ms: number; applied_quality: GraphicsQuality };
    }
  | { event: "ON_CHECKOUT_COUNTER_ENTERED"; payload: Record<string, never> };

export type UnityEventName = UnityEvent["event"];

// ---------------------------------------------------------------------------
// WebSocket (/ws/v1/session/{session_id}?token=...)
//
// Binary frames (client -> server): first byte 0x01, then PCM 16 kHz, 16-bit,
// mono, little-endian. A 20 ms frame carries 640 data bytes. Any other first
// byte is ignored by the server.
// ---------------------------------------------------------------------------

export const AUDIO_FRAME_PREFIX = 0x01;

interface WsEnvelope<T extends string, P> {
  type: T;
  /** Unix epoch milliseconds. */
  timestamp: number;
  payload: P;
}

export type EmptyPayload = Record<string, never>;

// Client -> Server

export type AudioEndMessage = WsEnvelope<"AUDIO_END", { audio_mode: "ptt" | "vad" }>;

export type TextMessage = WsEnvelope<
  "TEXT_MESSAGE",
  { text: string; language: Language; want_audio?: boolean }
>;

export type CvFrameRoiMessage = WsEnvelope<
  "CV_FRAME_ROI",
  {
    /** JPEG quality 0.7, <= 320x320 px, <= 64 KB decoded. */
    image_base64: string;
    hand_confidence: number;
    gaze_target_product_id: ProductId | null;
  }
>;

export type PlayerTelemetryMessage = WsEnvelope<"PLAYER_TELEMETRY", PlayerMovedPayload>;

export type BargeInInterruptMessage = WsEnvelope<"BARGE_IN_INTERRUPT", { response_id: string }>;

export type HeartbeatMessage = WsEnvelope<"HEARTBEAT", EmptyPayload>;

export type ResyncRequestMessage = WsEnvelope<"RESYNC_REQUEST", EmptyPayload>;

export type ClientMessage =
  | AudioEndMessage
  | TextMessage
  | CvFrameRoiMessage
  | PlayerTelemetryMessage
  | BargeInInterruptMessage
  | HeartbeatMessage
  | ResyncRequestMessage;

// Server -> Client

export type SessionSyncMessage = WsEnvelope<
  "SESSION_SYNC",
  {
    cart: CartSummary;
    spatial: { position: Vector3; current_aisle: string | null };
    last_waypoint_id: WaypointId | null;
  }
>;

export type SttTranscriptMessage = WsEnvelope<
  "STT_TRANSCRIPT",
  { text: string; confidence: number; is_final: boolean; language: Language }
>;

export interface AgentResponseChunk {
  response_id: string;
  chunk_seq: number;
  is_final: boolean;
  text: string;
  /** null for text-only replies. */
  audio_base64: string | null;
  audio_format: "mp3" | "opus" | null;
  audio_duration_ms: number;
  /** time_ms is relative to the start of this chunk; empty for text-only replies. */
  visemes: VisemeFrame[];
  /** May appear only on the final chunk. */
  recommended_products?: ProductId[];
  /** Carried by the first chunk. */
  avatar_animation?: AvatarAnimation;
}

export type AgentResponseStreamMessage = WsEnvelope<"AGENT_RESPONSE_STREAM", AgentResponseChunk>;

export type ResponseCancelledMessage = WsEnvelope<"RESPONSE_CANCELLED", { response_id: string }>;

export type UnityCommandMessage = WsEnvelope<
  "UNITY_COMMAND",
  { response_id: string; commands: UnityCommand[] }
>;

export type UiActionPayload =
  | { action: "OPEN_PRODUCT_INSPECTOR"; product_id: ProductId }
  | { action: "OPEN_CHECKOUT" }
  | { action: "CLOSE_MODAL" };

export type UiActionMessage = WsEnvelope<"UI_ACTION", UiActionPayload>;

export type CartUpdatedMessage = WsEnvelope<"CART_UPDATED", CartSummary>;

export type CvQualityWarningMessage = WsEnvelope<
  "CV_QUALITY_WARNING",
  { warning: QualityWarning; mean_luminance: number }
>;

export type WsErrorCode = "STT_FAILED" | "CV_UNSUPPORTED_IMAGE" | "RATE_LIMITED" | "INTERNAL";

export type WsErrorMessage = WsEnvelope<
  "ERROR",
  { code: WsErrorCode; message: string; recoverable: boolean }
>;

export type ServerMessage =
  | SessionSyncMessage
  | SttTranscriptMessage
  | AgentResponseStreamMessage
  | ResponseCancelledMessage
  | UnityCommandMessage
  | UiActionMessage
  | CartUpdatedMessage
  | CvQualityWarningMessage
  | HeartbeatMessage
  | WsErrorMessage;
