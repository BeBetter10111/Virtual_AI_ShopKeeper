"""Webcam ROI analysis: brightness check, dominant colour, CLIP embedding, category.

The CPU-heavy work runs in a worker thread (`VisionService.analyze`). The CLIP model
is loaded once by `VisionService.load()`, which the application should await at
startup. Optional environment variables for the default encoder:
    CLIP_MODEL_NAME   open_clip model name   (default "ViT-B-32", 512-d output)
    CLIP_PRETRAINED   open_clip weights tag  (default "laion2b_s34b_b79k")
    CLIP_NUM_THREADS  torch intra-op threads (default: torch's own default)
"""

import asyncio
import functools
import io
import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger("app.vision")

QualityWarning = Literal["LOW_LIGHT", "OVEREXPOSED"]

LOW_LIGHT_THRESHOLD = 40.0
OVEREXPOSED_THRESHOLD = 235.0
DOMAIN_CONFIDENCE_THRESHOLD = 0.45
EMBEDDING_DIM = 512

MAX_SIDE_PX = 320
MIN_SIDE_PX = 16
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_INPUT_PIXELS = 16_000_000

# Dominant-colour estimation.
SAMPLE_SIDE = 48
KMEANS_CLUSTERS = 3
KMEANS_ITERATIONS = 10
# Spread of the centre weighting, as a fraction of image width. At 0.15 an object about
# a third of the frame width wide still outweighs a plain background around it.
CENTER_SIGMA = 0.15
MERGE_DISTANCE = 12.0  # clusters this close to the winner (Lab) count as shading

Image8 = NDArray[np.uint8]
Vec = NDArray[np.float64]


class ImageDecodeError(ValueError):
    """The bytes are not a usable image."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSpec:
    """One zero-shot class. `category_slug` is None for out-of-domain objects."""

    label: str
    prompt: str
    category_slug: str | None


@dataclass(frozen=True)
class EncodedImage:
    embedding: list[float]  # L2-normalised, EMBEDDING_DIM values
    label_probs: dict[str, float]  # label -> probability, sums to 1


@dataclass(frozen=True)
class CategoryEstimate:
    label: str
    confidence: float
    is_in_domain: bool
    category_slug: str | None


@dataclass(frozen=True)
class VisionResult:
    lab: tuple[float, float, float]
    color_name: str
    mean_luminance: float
    quality_warning: QualityWarning | None
    embedding: list[float]
    category_label: str
    category_confidence: float
    is_in_domain: bool
    category_slug: str | None
    timings_ms: dict[str, float] = field(default_factory=dict)


# Labels must match the store's category slugs (see scripts/seed_store.py). When a
# category is added to the catalogue, add an in-domain LabelSpec for it here.
DEFAULT_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec("shirt", "a photo of a shirt", "shirts"),
    LabelSpec("jacket", "a photo of a jacket", "jackets"),
    LabelSpec("pants", "a photo of a pair of pants", "pants"),
    LabelSpec("shoes", "a photo of a pair of shoes", "shoes"),
    LabelSpec("mug", "a photo of a coffee mug", None),
    LabelSpec("phone", "a photo of a smartphone", None),
    LabelSpec("bottle", "a photo of a water bottle", None),
    LabelSpec("book", "a photo of a book", None),
    LabelSpec("laptop", "a photo of a laptop", None),
    LabelSpec("keys", "a photo of a bunch of keys", None),
    LabelSpec("pen", "a photo of a pen", None),
    LabelSpec("toy", "a photo of a toy", None),
    LabelSpec("plant", "a photo of a houseplant", None),
    LabelSpec("food", "a photo of a piece of food", None),
    LabelSpec("empty hand", "a photo of an empty hand", None),
    LabelSpec("wall", "a photo of a plain wall", None),
)


# ---------------------------------------------------------------------------
# Colour maths
# ---------------------------------------------------------------------------


def delta_e_2000(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005)."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.0

    d_lp = l2 - l1
    d_cp = c2p - c1p
    if c1p * c2p == 0:
        d_hp = 0.0
    else:
        diff = h2p - h1p
        d_hp = diff if abs(diff) <= 180 else diff - 360 if diff > 180 else diff + 360
    d_hh = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(d_hp / 2))

    l_bar = (l1 + l2) / 2
    c_bar_p = (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        h_bar = (h1p + h2p + 360) / 2
    else:
        h_bar = (h1p + h2p - 360) / 2

    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar))
        + 0.32 * math.cos(math.radians(3 * h_bar + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    )
    d_theta = 30 * math.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * math.sqrt(c_bar_p**7 / (c_bar_p**7 + 25**7))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_bar_p
    s_h = 1 + 0.015 * c_bar_p * t
    r_t = -math.sin(math.radians(2 * d_theta)) * r_c

    term_l = d_lp / s_l
    term_c = d_cp / s_c
    term_h = d_hh / s_h
    return math.sqrt(term_l**2 + term_c**2 + term_h**2 + r_t * term_c * term_h)


def lab_from_hex(hex_color: str) -> tuple[float, float, float]:
    """'#RRGGBB' (sRGB, D65) -> CIELAB, computed with the same OpenCV conversion."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    pixel = np.array([[[b, g, r]]], dtype=np.float32)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0, 0]
    return (float(lab[0]), float(lab[1]), float(lab[2]))


# Anchor colours used to name a measured colour. Entries that also appear in the
# seed catalogue use the same hex values so the names agree.
PALETTE: tuple[tuple[str, str], ...] = (
    ("Black", "#1A1A1A"),
    ("Dark Grey", "#4A4A4A"),
    ("Grey", "#808080"),
    ("Light Grey", "#C0C0C0"),
    ("White", "#F5F5F5"),
    ("Red", "#B22222"),
    ("Brick Red", "#A83232"),
    ("Burgundy", "#6D1F2E"),
    ("Orange", "#EF6C00"),
    ("Yellow", "#F9C80E"),
    ("Beige", "#C8B79A"),
    ("Khaki", "#B5A27A"),
    ("Brown", "#5C3A21"),
    ("Olive Green", "#6B7A3A"),
    ("Green", "#2E8B57"),
    ("Teal", "#1F7A7A"),
    ("Light Blue", "#8EC1E8"),
    ("Denim Blue", "#4F6D8F"),
    ("Blue", "#1E5BD8"),
    ("Navy Blue", "#1F3A5F"),
    ("Purple", "#6A3FA0"),
    ("Pink", "#E8869D"),
)


@functools.cache
def _palette_labs() -> tuple[tuple[str, tuple[float, float, float]], ...]:
    return tuple((name, lab_from_hex(hex_color)) for name, hex_color in PALETTE)


def name_color(lab: Sequence[float]) -> str:
    """Name of the palette colour closest to `lab` (by CIEDE2000)."""
    return min(_palette_labs(), key=lambda entry: delta_e_2000(lab, entry[1]))[0]


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------


def decode_image(data: bytes) -> Image8:
    """Decode JPEG/PNG bytes to a BGR array no larger than MAX_SIDE_PX."""
    if not data or len(data) > MAX_INPUT_BYTES:
        raise ImageDecodeError("Image is empty or too large.")
    try:
        # Reading only the header lets us reject decompression bombs cheaply.
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
    except Exception as exc:
        raise ImageDecodeError("Not a decodable image.") from exc
    if width * height > MAX_INPUT_PIXELS:
        raise ImageDecodeError("Image dimensions are too large.")

    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ImageDecodeError("Not a decodable image.")
    bgr: Image8 = np.asarray(decoded, dtype=np.uint8)

    height, width = bgr.shape[:2]
    if min(height, width) < MIN_SIDE_PX:
        raise ImageDecodeError("Image is too small.")
    longest = max(height, width)
    if longest > MAX_SIDE_PX:
        scale = MAX_SIDE_PX / longest
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        bgr = np.asarray(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA))
    return bgr


def mean_luminance(bgr: Image8) -> float:
    """Mean of the grey-level image, 0-255."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def quality_warning_for(mean: float) -> QualityWarning | None:
    if mean < LOW_LIGHT_THRESHOLD:
        return "LOW_LIGHT"
    if mean > OVEREXPOSED_THRESHOLD:
        return "OVEREXPOSED"
    return None


@functools.cache
def _center_weights(side: int) -> Vec:
    coords = (np.arange(side) + 0.5) / side - 0.5
    xx, yy = np.meshgrid(coords, coords)
    weights: Vec = np.exp(-(xx**2 + yy**2) / (2 * CENTER_SIGMA**2)).reshape(-1)
    weights.setflags(write=False)
    return weights


def _weighted_kmeans(
    points: Vec, weights: Vec, k: int, iterations: int
) -> tuple[Vec, NDArray[np.intp]]:
    """Deterministic weighted k-means (k-means++ seeding with a fixed seed)."""
    rng = np.random.default_rng(0)
    count = len(points)
    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    centers[0] = points[rng.choice(count, p=weights / weights.sum())]
    for i in range(1, k):
        dist = ((points[:, None, :] - centers[None, :i, :]) ** 2).sum(axis=2)
        score = dist.min(axis=1) * weights
        total = score.sum()
        if total <= 1e-9:
            centers[i] = points[rng.integers(count)]
        else:
            centers[i] = points[rng.choice(count, p=score / total)]

    labels = np.zeros(count, dtype=np.intp)
    for _ in range(iterations):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for j in range(k):
            members = labels == j
            if members.any():
                centers[j] = np.average(
                    points[members], axis=0, weights=weights[members]
                )
    return centers, labels


def dominant_lab(bgr: Image8) -> tuple[float, float, float]:
    """Dominant colour of the object in the frame, as CIELAB.

    The object is assumed to be near the centre (the browser crops the ROI around the
    hand), so pixels are weighted by a Gaussian centred on the frame. Pixels are
    clustered in Lab; the cluster with the most weight wins, together with clusters so
    close to it that they are just shading on the same material.
    """
    small = cv2.resize(bgr, (SAMPLE_SIDE, SAMPLE_SIDE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    points: Vec = lab.reshape(-1, 3).astype(np.float64)
    weights = _center_weights(SAMPLE_SIDE)

    centers, labels = _weighted_kmeans(
        points, weights, KMEANS_CLUSTERS, KMEANS_ITERATIONS
    )
    mass = np.bincount(labels, weights=weights, minlength=KMEANS_CLUSTERS)
    winner = int(mass.argmax())

    close = np.linalg.norm(centers - centers[winner], axis=1) < MERGE_DISTANCE
    merged = np.flatnonzero(close & (mass > 0))  # always contains the winner
    selected = np.isin(labels, merged)
    result = np.average(points[selected], axis=0, weights=weights[selected])
    return (float(result[0]), float(result[1]), float(result[2]))


# ---------------------------------------------------------------------------
# Category estimate
# ---------------------------------------------------------------------------


def estimate_category(
    label_probs: dict[str, float],
    labels: Sequence[LabelSpec] = DEFAULT_LABELS,
    threshold: float = DOMAIN_CONFIDENCE_THRESHOLD,
) -> CategoryEstimate:
    """Turn zero-shot probabilities into a category decision.

    `confidence` is the total probability that the object is store merchandise (the
    sum over in-domain labels). When it reaches `threshold` the label is the most
    likely store category; otherwise it names the most likely non-store object.
    """
    known = {spec.label: max(0.0, label_probs.get(spec.label, 0.0)) for spec in labels}
    total = sum(known.values())
    if total <= 0:
        return CategoryEstimate("unknown", 0.0, False, None)
    probs = {label: p / total for label, p in known.items()}

    in_domain = [s for s in labels if s.category_slug is not None]
    out_domain = [s for s in labels if s.category_slug is None]
    confidence = min(1.0, sum(probs[s.label] for s in in_domain))

    if confidence >= threshold and in_domain:
        best = max(in_domain, key=lambda s: probs[s.label])
        return CategoryEstimate(best.label, confidence, True, best.category_slug)
    if out_domain:
        other = max(out_domain, key=lambda s: probs[s.label])
        return CategoryEstimate(other.label, confidence, False, None)
    return CategoryEstimate("unknown", confidence, False, None)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class ImageEncoder(Protocol):
    def load(self) -> None: ...

    def encode(self, rgb: Image8) -> EncodedImage: ...


class ClipEncoder:
    """CLIP image encoder with zero-shot label probabilities (open_clip, CPU)."""

    def __init__(
        self,
        labels: Sequence[LabelSpec] = DEFAULT_LABELS,
        model_name: str | None = None,
        pretrained: str | None = None,
    ) -> None:
        self._labels = tuple(labels)
        self._model_name = model_name or os.environ.get("CLIP_MODEL_NAME", "ViT-B-32")
        self._pretrained = pretrained or os.environ.get(
            "CLIP_PRETRAINED", "laion2b_s34b_b79k"
        )
        self._model: Any = None
        self._preprocess: Any = None
        self._text_features: Any = None
        self._logit_scale = 100.0

    def load(self) -> None:
        import open_clip
        import torch

        threads = os.environ.get("CLIP_NUM_THREADS")
        if threads:
            torch.set_num_threads(int(threads))

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        model.eval()
        if int(model.visual.output_dim) != EMBEDDING_DIM:
            raise RuntimeError(
                f"{self._model_name} produces {model.visual.output_dim}-d embeddings; "
                f"the database column expects {EMBEDDING_DIM}."
            )
        tokenizer = open_clip.get_tokenizer(self._model_name)
        with torch.inference_mode():
            tokens = tokenizer([spec.prompt for spec in self._labels])
            text = model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)

        self._model, self._preprocess, self._text_features = model, preprocess, text
        self._logit_scale = float(model.logit_scale.exp().item())
        logger.info("Loaded CLIP %s (%s)", self._model_name, self._pretrained)

    def encode(self, rgb: Image8) -> EncodedImage:
        if self._model is None or self._preprocess is None:
            raise RuntimeError("CLIP model is not loaded; call load() first.")
        import torch

        tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0)
        with torch.inference_mode():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = self._logit_scale * features @ self._text_features.T
            probs = logits.softmax(dim=-1)[0]
        return EncodedImage(
            embedding=[float(v) for v in features[0].tolist()],
            label_probs={
                spec.label: float(p) for spec, p in zip(self._labels, probs.tolist())
            },
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class VisionService:
    def __init__(
        self,
        encoder: ImageEncoder,
        labels: Sequence[LabelSpec] = DEFAULT_LABELS,
        max_concurrency: int = 2,
    ) -> None:
        self._encoder = encoder
        self._labels = tuple(labels)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def load(self) -> None:
        """Load the model once (call at application startup)."""
        await asyncio.to_thread(self._encoder.load)

    def analyze_sync(self, data: bytes) -> VisionResult:
        started = time.perf_counter()
        bgr = decode_image(data)
        decoded = time.perf_counter()

        mean = mean_luminance(bgr)
        lab = dominant_lab(bgr)
        color_done = time.perf_counter()

        encoded = self._encoder.encode(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        encoded_done = time.perf_counter()
        if len(encoded.embedding) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Encoder returned {len(encoded.embedding)} values, "
                f"expected {EMBEDDING_DIM}."
            )
        category = estimate_category(encoded.label_probs, self._labels)

        timings = {
            "decode": (decoded - started) * 1000,
            "color": (color_done - decoded) * 1000,
            "encode": (encoded_done - color_done) * 1000,
            "total": (time.perf_counter() - started) * 1000,
        }
        logger.debug("vision timings ms: %s", timings)
        return VisionResult(
            lab=lab,
            color_name=name_color(lab),
            mean_luminance=mean,
            quality_warning=quality_warning_for(mean),
            embedding=encoded.embedding,
            category_label=category.label,
            category_confidence=category.confidence,
            is_in_domain=category.is_in_domain,
            category_slug=category.category_slug,
            timings_ms=timings,
        )

    async def analyze(self, data: bytes) -> VisionResult:
        """Analyze one ROI frame without blocking the event loop."""
        async with self._semaphore:
            return await asyncio.to_thread(self.analyze_sync, data)


@functools.cache
def get_vision_service() -> VisionService:
    """Process-wide service with the default CLIP encoder (call `load()` once)."""
    return VisionService(ClipEncoder())
