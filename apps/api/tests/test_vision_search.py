import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import pytest_asyncio
from numpy.typing import NDArray
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.models.domain import Product, ProductVariant
from app.schemas import contracts as c
from app.services import inventory_search as search
from app.services import vision_service as vision
from scripts.seed_store import PRODUCTS, seed

RED = "#B22222"  # "Red Flannel Shirt" (prod_003, sold out in every size)
NAVY = "#1F3A5F"  # "Blue Oxford Shirt" (prod_001, size L sold out only)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bgr(hex_color: str) -> tuple[int, int, int]:
    return (int(hex_color[5:7], 16), int(hex_color[3:5], 16), int(hex_color[1:3], 16))


def solid(hex_color: str, width: int = 320, height: int = 320) -> NDArray[np.uint8]:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bgr(hex_color)
    return frame


def gray(level: int, size: int = 64) -> NDArray[np.uint8]:
    return np.full((size, size, 3), level, dtype=np.uint8)


def as_png(frame: NDArray[np.uint8]) -> bytes:
    return cv2.imencode(".png", frame)[1].tobytes()


def as_jpeg(frame: NDArray[np.uint8], quality: int = 70) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    return cv2.imencode(".jpg", frame, params)[1].tobytes()


def object_on_background(
    object_hex: str, side: int, background_hex: str = "#F5F5F5"
) -> NDArray[np.uint8]:
    frame = solid(background_hex)
    start = (320 - side) // 2
    frame[start : start + side, start : start + side] = bgr(object_hex)
    return frame


def unit_vector(index: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(vision.EMBEDDING_DIM)]


def probs_for(**weights: float) -> dict[str, float]:
    """Label probabilities with the given weights; the rest is spread evenly."""
    labels = [spec.label for spec in vision.DEFAULT_LABELS]
    given = {k.replace("_", " "): v for k, v in weights.items()}
    rest = [label for label in labels if label not in given]
    share = max(0.0, 1.0 - sum(given.values())) / len(rest)
    return {**{label: share for label in rest}, **given}


class FakeEncoder:
    """Scripted stand-in for CLIP: fixed probabilities, records the calling thread."""

    def __init__(self, label_probs: dict[str, float], embedding: list[float]) -> None:
        self.label_probs = label_probs
        self.embedding = embedding
        self.loaded = False
        self.threads: list[str] = []

    def load(self) -> None:
        self.loaded = True
        self.threads.append(threading.current_thread().name)

    def encode(self, rgb: NDArray[np.uint8]) -> vision.EncodedImage:
        self.threads.append(threading.current_thread().name)
        return vision.EncodedImage(self.embedding, self.label_probs)


def service_with(
    label_probs: dict[str, float] | None = None, embedding: list[float] | None = None
) -> tuple[vision.VisionService, FakeEncoder]:
    encoder = FakeEncoder(
        label_probs or probs_for(shirt=0.9), embedding or unit_vector(0)
    )
    return vision.VisionService(encoder), encoder


# ---------------------------------------------------------------------------
# Colour distance and naming
# ---------------------------------------------------------------------------

# Published CIEDE2000 test pairs (Sharma, Wu & Dalal, 2005).
SHARMA_PAIRS = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((50.0, 2.5, 0.0), (58.0, 24.0, 15.0), 19.4535),
]


@pytest.mark.parametrize(("lab1", "lab2", "expected"), SHARMA_PAIRS)
def test_delta_e_2000_matches_published_values(
    lab1: tuple[float, ...], lab2: tuple[float, ...], expected: float
) -> None:
    assert vision.delta_e_2000(lab1, lab2) == pytest.approx(expected, abs=5e-4)
    assert vision.delta_e_2000(lab2, lab1) == pytest.approx(expected, abs=5e-4)


def test_delta_e_2000_is_zero_for_identical_colours() -> None:
    assert vision.delta_e_2000((40.0, 10.0, -20.0), (40.0, 10.0, -20.0)) == 0.0


@pytest.mark.parametrize("spec", PRODUCTS, ids=lambda p: p.public_id)
def test_colour_names_agree_with_the_catalogue(spec: Any) -> None:
    lab = vision.lab_from_hex(spec.color_hex)
    assert vision.name_color(lab) == spec.color_name


# ---------------------------------------------------------------------------
# Brightness check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "warning"),
    [
        (0, "LOW_LIGHT"),
        (39, "LOW_LIGHT"),
        (40, None),
        (128, None),
        (235, None),
        (236, "OVEREXPOSED"),
        (255, "OVEREXPOSED"),
    ],
)
def test_quality_warning_thresholds(level: int, warning: str | None) -> None:
    mean = vision.mean_luminance(vision.decode_image(as_png(gray(level))))
    assert mean == pytest.approx(level, abs=0.5)
    assert vision.quality_warning_for(mean) == warning


@pytest.mark.parametrize(
    ("level", "warning"), [(15, "LOW_LIGHT"), (128, None), (250, "OVEREXPOSED")]
)
def test_service_reports_quality_warning_for_jpeg_frames(
    level: int, warning: str | None
) -> None:
    service, _ = service_with()
    result = service.analyze_sync(as_jpeg(gray(level, 320)))
    assert result.quality_warning == warning
    assert result.mean_luminance == pytest.approx(level, abs=3)


# ---------------------------------------------------------------------------
# Dominant colour
# ---------------------------------------------------------------------------


def dominant_delta_e(frame: NDArray[np.uint8], expected_hex: str) -> float:
    lab = vision.dominant_lab(vision.decode_image(as_jpeg(frame)))
    return vision.delta_e_2000(lab, vision.lab_from_hex(expected_hex))


def test_solid_colour_is_recovered_and_named() -> None:
    lab = vision.dominant_lab(vision.decode_image(as_jpeg(solid(NAVY))))
    assert vision.delta_e_2000(lab, vision.lab_from_hex(NAVY)) < 2
    assert vision.name_color(lab) == "Navy Blue"


@pytest.mark.parametrize("side", [112, 160, 200])
def test_centred_object_wins_over_a_larger_plain_background(side: int) -> None:
    frame = object_on_background(RED, side)
    assert dominant_delta_e(frame, RED) < 3


def test_lighting_gradient_does_not_shift_the_colour_much() -> None:
    base = np.array(bgr(NAVY), dtype=np.float32)
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    for y in range(320):
        frame[y, :] = np.clip(base * (0.8 + 0.4 * y / 320), 0, 255)
    assert dominant_delta_e(frame, NAVY) < 4


def test_object_on_cluttered_background() -> None:
    rng = np.random.default_rng(1)
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    for y in range(0, 320, 40):
        for x in range(0, 320, 40):
            frame[y : y + 40, x : x + 40] = rng.integers(0, 255, 3)
    frame[70:250, 70:250] = bgr("#B5A27A")
    assert dominant_delta_e(frame, "#B5A27A") < 4


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data", [b"", b"not an image", b"\xff\xd8\xff" + b"\x00" * 50])
def test_undecodable_bytes_are_rejected(data: bytes) -> None:
    with pytest.raises(vision.ImageDecodeError):
        vision.decode_image(data)


def test_tiny_image_is_rejected() -> None:
    with pytest.raises(vision.ImageDecodeError):
        vision.decode_image(as_png(gray(128, 8)))


def test_image_with_too_many_pixels_is_rejected_before_decoding() -> None:
    huge = np.zeros((5000, 5000, 3), dtype=np.uint8)  # 25M pixels, tiny as PNG
    with pytest.raises(vision.ImageDecodeError):
        vision.decode_image(as_png(huge))


def test_oversized_upload_is_rejected() -> None:
    with pytest.raises(vision.ImageDecodeError):
        vision.decode_image(b"\x00" * (vision.MAX_INPUT_BYTES + 1))


def test_large_frames_are_downscaled() -> None:
    frame = vision.decode_image(as_jpeg(solid(NAVY, 1000, 800)))
    assert max(frame.shape[:2]) == vision.MAX_SIDE_PX
    assert frame.shape[0] < frame.shape[1]  # aspect ratio kept


# ---------------------------------------------------------------------------
# Category decision
# ---------------------------------------------------------------------------


def test_clear_shirt_is_in_domain() -> None:
    estimate = vision.estimate_category(probs_for(shirt=0.88))
    assert (estimate.label, estimate.is_in_domain) == ("shirt", True)
    assert estimate.category_slug == "shirts"
    # The shirt's 0.88 plus the small shares of the other store labels.
    assert estimate.confidence == pytest.approx(0.90, abs=0.02)


@pytest.mark.parametrize("object_label", ["mug", "phone"])
def test_cup_and_phone_are_out_of_domain(object_label: str) -> None:
    estimate = vision.estimate_category(probs_for(**{object_label: 0.9}))
    assert estimate.is_in_domain is False
    assert estimate.label == object_label
    assert estimate.category_slug is None
    assert estimate.confidence < vision.DOMAIN_CONFIDENCE_THRESHOLD


def test_threshold_is_inclusive_at_0_45() -> None:
    def estimate_for(in_domain_mass: float) -> vision.CategoryEstimate:
        probs = {spec.label: 0.0 for spec in vision.DEFAULT_LABELS}
        probs["shirt"] = in_domain_mass
        probs["mug"] = 1.0 - in_domain_mass
        return vision.estimate_category(probs)

    assert estimate_for(0.45).is_in_domain is True
    assert estimate_for(0.449).is_in_domain is False


def test_probability_is_pooled_over_all_store_categories() -> None:
    # No single store label is the top one, but together they pass the threshold.
    probs = probs_for(shirt=0.2, jacket=0.2, pants=0.1, mug=0.3)
    estimate = vision.estimate_category(probs)
    assert estimate.is_in_domain is True
    assert estimate.label in {"shirt", "jacket"}
    assert estimate.confidence == pytest.approx(0.5, abs=0.05)


def test_empty_probabilities_are_not_in_domain() -> None:
    assert vision.estimate_category({}).is_in_domain is False


# ---------------------------------------------------------------------------
# VisionService
# ---------------------------------------------------------------------------


def test_service_combines_colour_brightness_and_category() -> None:
    service, _ = service_with(probs_for(shirt=0.9), embedding=unit_vector(7))
    result = service.analyze_sync(as_jpeg(object_on_background(NAVY, 200)))

    assert result.color_name == "Navy Blue"
    assert result.quality_warning is None
    assert (result.category_label, result.is_in_domain) == ("shirt", True)
    assert result.category_slug == "shirts"
    assert result.embedding == unit_vector(7)
    assert set(result.timings_ms) == {"decode", "color", "encode", "total"}


def test_service_flags_a_mug_as_out_of_domain() -> None:
    service, _ = service_with(probs_for(mug=0.92))
    result = service.analyze_sync(as_jpeg(solid("#FFFFFF")))
    assert result.is_in_domain is False
    assert result.category_label == "mug"
    assert result.category_slug is None


def test_service_rejects_embeddings_of_the_wrong_size() -> None:
    service, _ = service_with(embedding=[0.1] * 10)
    with pytest.raises(RuntimeError, match="expected 512"):
        service.analyze_sync(as_jpeg(solid(NAVY)))


async def test_analysis_runs_off_the_event_loop_thread() -> None:
    service, encoder = service_with()
    await service.load()
    result = await service.analyze(as_jpeg(solid(NAVY)))

    assert encoder.loaded
    assert result.color_name == "Navy Blue"
    assert threading.current_thread().name not in encoder.threads


def test_colour_pipeline_is_far_inside_the_350_ms_budget() -> None:
    # Everything except the CLIP forward pass, on a 320x320 JPEG frame.
    service, _ = service_with()
    data = as_jpeg(object_on_background(NAVY, 200))
    service.analyze_sync(data)  # warm-up

    durations = []
    for _ in range(5):
        started = time.perf_counter()
        service.analyze_sync(data)
        durations.append((time.perf_counter() - started) * 1000)
    assert sorted(durations)[2] < 350


@pytest.mark.skipif(
    os.environ.get("RUN_CLIP_TESTS") != "1",
    reason="downloads CLIP weights; run with RUN_CLIP_TESTS=1",
)
def test_real_clip_encoder_is_512d_and_fast_enough() -> None:
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    encoder = vision.ClipEncoder()
    encoder.load()
    service = vision.VisionService(encoder)
    data = as_jpeg(object_on_background(NAVY, 200))

    service.analyze_sync(data)  # warm-up
    totals = [service.analyze_sync(data).timings_ms["total"] for _ in range(5)]
    median = sorted(totals)[2]
    print(f"\nreal CLIP, 320x320 frame: median {median:.0f} ms (budget 350 ms)")

    encoded = encoder.encode(cv2.cvtColor(solid(NAVY), cv2.COLOR_BGR2RGB))
    assert len(encoded.embedding) == vision.EMBEDDING_DIM
    assert np.linalg.norm(encoded.embedding) == pytest.approx(1.0, abs=1e-3)
    assert sum(encoded.label_probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert median < 350


# ---------------------------------------------------------------------------
# Scoring and alternatives (no database)
# ---------------------------------------------------------------------------


def scored(
    public_id: str, delta_e: float | None, in_stock: bool = True, score: float = 0.5
) -> search.ScoredProduct:
    product = SimpleNamespace(id=public_id, public_id=public_id)
    return search.ScoredProduct(
        product=product,
        delta_e00=delta_e,
        similarity=0.0,
        score=score,
        in_stock=in_stock,
    )


def test_combine_score() -> None:
    assert search.combine_score(0.0, 1.0, use_embedding=True) == pytest.approx(1.0)
    assert search.combine_score(0.0, 0.0, use_embedding=True) == pytest.approx(0.7)
    assert search.combine_score(25.0, 0.5, use_embedding=True) == pytest.approx(0.5)
    assert search.combine_score(25.0, 0.9, use_embedding=False) == pytest.approx(0.5)
    assert search.combine_score(80.0, 0.0, use_embedding=False) == 0.0
    assert search.combine_score(None, 0.8, use_embedding=True) == 0.8


def test_no_alternatives_when_the_best_match_is_in_stock() -> None:
    best = scored("best", 1.0, in_stock=True)
    assert search.pick_alternatives([best, scored("other", 2.0)], best) == []


def test_alternatives_are_in_stock_close_and_capped_at_two() -> None:
    best = scored("best", 0.5, in_stock=False)
    ranked = [
        best,
        scored("c", 9.0),
        scored("a", 2.0),
        scored("sold_out", 1.0, in_stock=False),
        scored("b", 4.0),
        scored("far", 25.0),
    ]
    picked = search.pick_alternatives(ranked, best)
    assert [p.product.public_id for p in picked] == ["a", "b"]


def test_alternative_distance_limit_is_strictly_below_10() -> None:
    best = scored("best", 0.5, in_stock=False)
    assert search.pick_alternatives([best, scored("edge", 10.0)], best) == []
    just_under = scored("under", 9.99)
    assert search.pick_alternatives([best, just_under], best) == [just_under]


# ---------------------------------------------------------------------------
# Search against the database
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def seeded(db_engine: AsyncEngine) -> None:
    await seed(db_engine)


async def neutralize_embeddings(db: AsyncSession) -> None:
    """Give every product the same embedding so only colour decides the ranking."""
    await db.execute(update(Product).values(image_embedding=unit_vector(0)))
    await db.commit()


def make_analysis(
    hex_color: str,
    *,
    slug: str | None = "shirts",
    embedding: list[float] | None = None,
    in_domain: bool = True,
    warning: vision.QualityWarning | None = None,
    label: str = "shirt",
) -> vision.VisionResult:
    lab = vision.lab_from_hex(hex_color)
    return vision.VisionResult(
        lab=lab,
        color_name=vision.name_color(lab),
        mean_luminance=128.0,
        quality_warning=warning,
        embedding=embedding or unit_vector(0),
        category_label=label,
        category_confidence=0.9 if in_domain else 0.05,
        is_in_domain=in_domain,
        category_slug=slug if in_domain else None,
    )


async def test_exact_colour_match_in_stock_has_no_alternatives(
    db_session: AsyncSession, seeded: None
) -> None:
    await neutralize_embeddings(db_session)
    response = await search.visual_search(db_session, make_analysis(NAVY))

    assert c.VisualSearchResponse.model_validate(response.model_dump()) == response
    best = response.matches[0]
    assert best.product_id == "prod_001"
    assert best.waypoint_id == "WP_AISLE_01_SHIRTS"
    assert best.delta_e00 < 1
    assert best.in_stock is True  # size L is sold out, the other sizes are not
    assert response.alternatives == []
    assert response.detected_color_name == "Navy Blue"
    scores = [m.similarity_score for m in response.matches]
    assert scores == sorted(scores, reverse=True)
    assert {m.product_id for m in response.matches} == {
        "prod_001",
        "prod_002",
        "prod_003",
        "prod_004",
    }


async def test_sold_out_best_match_offers_a_close_in_stock_alternative(
    db_session: AsyncSession, seeded: None
) -> None:
    await neutralize_embeddings(db_session)
    response = await search.visual_search(db_session, make_analysis(RED))

    best = response.matches[0]
    assert (best.product_id, best.in_stock) == ("prod_003", False)
    (alternative,) = response.alternatives
    assert alternative.product_id == "prod_004"
    assert alternative.in_stock is True
    assert alternative.delta_e00 < 10


async def test_alternatives_are_capped_at_two_and_stay_in_category(
    db_session: AsyncSession, seeded: None
) -> None:
    await neutralize_embeddings(db_session)
    red = vision.lab_from_hex(RED)
    for public_id, offset in (("prod_001", 1), ("prod_002", 2), ("prod_004", 3)):
        await db_session.execute(
            update(Product)
            .where(Product.public_id == public_id)
            .values(color_lab=[red[0] + offset, red[1], red[2]])
        )
    # An exact colour match, but a jacket: must never be offered for a shirt.
    await db_session.execute(
        update(Product)
        .where(Product.public_id == "prod_005")
        .values(color_lab=list(red))
    )
    await db_session.commit()

    response = await search.visual_search(db_session, make_analysis(RED))
    assert response.matches[0].product_id == "prod_003"
    assert [a.product_id for a in response.alternatives] == ["prod_001", "prod_002"]
    assert "prod_005" not in {m.product_id for m in response.matches}


async def test_distant_colours_are_not_offered_as_alternatives(
    db_session: AsyncSession, seeded: None
) -> None:
    await neutralize_embeddings(db_session)
    await db_session.execute(
        update(ProductVariant)
        .where(ProductVariant.public_id.like("var_001_%"))
        .values(stock_qty=0)
    )
    await db_session.commit()

    response = await search.visual_search(db_session, make_analysis(NAVY))
    assert (response.matches[0].product_id, response.matches[0].in_stock) == (
        "prod_001",
        False,
    )
    assert response.alternatives == []  # every other shirt is far from navy


async def test_out_of_domain_object_gets_no_matches(
    db_session: AsyncSession, seeded: None
) -> None:
    analysis = make_analysis(
        "#FFFFFF", in_domain=False, label="mug", warning="LOW_LIGHT"
    )
    response = await search.visual_search(db_session, analysis)

    assert c.VisualSearchResponse.model_validate(response.model_dump()) == response
    assert response.is_in_domain is False
    assert response.category_label == "mug"
    assert response.quality_warning == "LOW_LIGHT"
    assert response.matches == []
    assert response.alternatives == []
    assert response.detected_color_name == "White"


async def test_results_are_limited_to_the_detected_category(
    db_session: AsyncSession, seeded: None
) -> None:
    await neutralize_embeddings(db_session)
    black = "#1A1A1A"

    jackets = await search.visual_search(
        db_session, make_analysis(black, slug="jackets", label="jacket")
    )
    assert jackets.matches[0].product_id == "prod_005"
    jacket_ids = {m.product_id for m in jackets.matches}
    assert jacket_ids == {"prod_005", "prod_006", "prod_007"}

    shirts = await search.visual_search(db_session, make_analysis(black))
    assert {m.product_id for m in shirts.matches} == {
        "prod_001",
        "prod_002",
        "prod_003",
        "prod_004",
    }


async def test_match_count_is_capped(
    db_session: AsyncSession, seeded: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await neutralize_embeddings(db_session)
    monkeypatch.setattr(search, "MAX_MATCHES", 2)
    response = await search.visual_search(db_session, make_analysis(NAVY))
    assert len(response.matches) == 2


async def test_embedding_similarity_breaks_colour_ties(
    db_session: AsyncSession, seeded: None
) -> None:
    await neutralize_embeddings(db_session)
    navy = list(vision.lab_from_hex(NAVY))
    await db_session.execute(
        update(Product)
        .where(Product.public_id.in_(["prod_001", "prod_002"]))
        .values(color_lab=navy)
    )
    await db_session.execute(
        update(Product)
        .where(Product.public_id == "prod_002")
        .values(image_embedding=unit_vector(1))
    )
    await db_session.commit()

    analysis = make_analysis(NAVY, embedding=unit_vector(1))
    response = await search.visual_search(db_session, analysis)

    assert response.matches[0].product_id == "prod_002"
    by_id = {m.product_id: m for m in response.matches}
    assert by_id["prod_002"].similarity_score > by_id["prod_001"].similarity_score
    assert by_id["prod_002"].similarity_score == pytest.approx(1.0, abs=0.01)
    assert by_id["prod_001"].similarity_score == pytest.approx(0.7, abs=0.01)


async def test_rank_products_by_colour_only(
    db_session: AsyncSession, seeded: None
) -> None:
    lab = vision.lab_from_hex(NAVY)
    ranked = await search.rank_products(db_session, lab=lab)

    assert len(ranked) == 12
    assert ranked[0].product.public_id == "prod_001"
    assert ranked[0].similarity == 0.0
    top = ranked[0]
    assert top.score == pytest.approx(search.color_score(top.delta_e00 or 0))
    assert [s.score for s in ranked] == sorted((s.score for s in ranked), reverse=True)


async def test_rank_products_text_filter(
    db_session: AsyncSession, seeded: None
) -> None:
    async def ids(text: str) -> list[str]:
        ranked = await search.rank_products(db_session, query_text=text)
        return sorted(s.product.public_id for s in ranked)

    assert await ids("jacket") == ["prod_005", "prod_006"]
    assert await ids("DENIM jacket") == ["prod_006"]  # every word must match
    assert await ids("linen") == ["prod_002"]
    assert await ids("%") == []  # wildcards in user text are escaped
    assert len(await ids("")) == 12
