import asyncio
import json
import random
import threading
import time
import typing
from collections.abc import AsyncIterator
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.schemas import contracts as c
from app.services import speech_service as speech
from app.services.speech_service import (
    AzureSpeechProvider,
    MockSpeechProvider,
    SpeechError,
    SpeechService,
    SttAggregator,
    SttSegment,
)
from app.utils import viseme_mapper as vm
from app.utils.viseme_mapper import VisemeEvent

CANCEL_BUDGET_S = 0.1  # barge-in must stop work in under 100 ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_valid_timeline(events: list[VisemeEvent], duration_ms: int) -> None:
    assert events, "timeline must not be empty"
    assert events[0].time_ms == 0
    assert all(e.viseme in vm.OCULUS_VISEMES for e in events)
    times = [e.time_ms for e in events]
    assert all(isinstance(t, int) and t >= 0 for t in times)
    assert all(a < b for a, b in zip(times, times[1:], strict=False)), times
    assert times[-1] <= duration_ms


async def collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in stream]


async def pcm_frames(
    count: int, size: int = speech.PCM_FRAME_BYTES
) -> AsyncIterator[bytes]:
    for _ in range(count):
        yield b"\x01\x00" * (size // 2)


# ---------------------------------------------------------------------------
# Viseme mapper
# ---------------------------------------------------------------------------


def test_oculus_names_match_the_shared_contract() -> None:
    assert vm.OCULUS_VISEMES == typing.get_args(c.Viseme)
    assert len(vm.OCULUS_VISEMES) == 15


@pytest.mark.parametrize("viseme_id", range(22))
def test_every_azure_viseme_id_maps_to_an_oculus_name(viseme_id: int) -> None:
    assert vm.map_azure_viseme(viseme_id) in vm.OCULUS_VISEMES


@pytest.mark.parametrize(
    ("viseme_id", "expected"),
    [
        (0, "sil"),
        (21, "PP"),
        (18, "FF"),
        (15, "SS"),
        (20, "kk"),
        (14, "nn"),
        (99, "sil"),
    ],
)
def test_azure_viseme_spot_checks(viseme_id: int, expected: str) -> None:
    assert vm.map_azure_viseme(viseme_id) == expected


def test_from_azure_events() -> None:
    events = vm.from_azure_events([(0, 0), (120, 2), (300, 21)])
    assert events == [
        VisemeEvent(0, "sil"),
        VisemeEvent(120, "aa"),
        VisemeEvent(300, "PP"),
    ]


def test_sanitize_orders_clamps_and_merges() -> None:
    messy = [
        VisemeEvent(500, "aa"),
        VisemeEvent(-5, "zz"),  # unknown name, negative time
        VisemeEvent(500, "O"),  # same time: the later one wins
        VisemeEvent(9999, "PP"),  # beyond the audio
    ]
    result = vm.sanitize_visemes(messy, 1000)
    assert result == [
        VisemeEvent(0, "sil"),
        VisemeEvent(500, "O"),
        VisemeEvent(1000, "PP"),
    ]
    assert_valid_timeline(result, 1000)


def test_sanitize_inserts_leading_silence_and_closes_the_mouth() -> None:
    result = vm.sanitize_visemes([VisemeEvent(200, "aa")], 600)
    assert result == [
        VisemeEvent(0, "sil"),
        VisemeEvent(200, "aa"),
        VisemeEvent(600, "sil"),
    ]


def test_sanitize_merges_repeated_visemes() -> None:
    events = [VisemeEvent(0, "aa"), VisemeEvent(100, "aa"), VisemeEvent(200, "O")]
    assert [e.viseme for e in vm.sanitize_visemes(events, 500)] == ["aa", "O", "sil"]


@pytest.mark.parametrize("duration", [0, -10])
def test_sanitize_with_no_audio_gives_a_single_silence(duration: int) -> None:
    result = vm.sanitize_visemes([VisemeEvent(5, "aa")], duration)
    assert result == [VisemeEvent(0, "sil")]


def test_sanitize_of_nothing_is_silence() -> None:
    assert vm.sanitize_visemes([], 300) == [VisemeEvent(0, "sil")]


def names(text: str, language: str, duration: int = 3000) -> list[str]:
    return [e.viseme for e in vm.text_to_visemes(text, language, duration)]


def test_vietnamese_fallback_has_the_expected_mouth_shapes() -> None:
    shapes = names("Xin chào", "vi")
    assert shapes[:6] == ["SS", "I", "nn", "CH", "aa", "O"]
    assert "FF" in names("phở", "vi")
    assert names("ma", "vi") == ["PP", "aa", "sil"]
    assert names("nghe", "vi")[:2] == ["kk", "E"]


def test_english_fallback_has_the_expected_mouth_shapes() -> None:
    assert names("mama", "en") == ["PP", "aa", "PP", "aa", "sil"]
    assert names("fish", "en")[:2] == ["FF", "I"]
    assert "TH" in names("the", "en")
    assert names("city", "en")[0] == "SS"  # soft c
    assert names("cat", "en")[0] == "kk"  # hard c


def test_tone_marks_do_not_change_the_mouth_shapes() -> None:
    variants = ["ma", "má", "mà", "mả", "mã", "mạ"]
    assert len({tuple(names(word, "vi")) for word in variants}) == 1


def test_punctuation_closes_the_mouth() -> None:
    shapes = names("xin chào, bạn", "vi")
    assert "sil" in shapes[1:-1]


@pytest.mark.parametrize("text", ["", "   ", "1234 %$#", "😀", "...", ",,,"])
def test_text_without_speech_is_a_single_silence(text: str) -> None:
    assert vm.text_to_visemes(text, "vi", 1000) == [VisemeEvent(0, "sil")]


def test_fallback_is_valid_for_arbitrary_text_and_durations() -> None:
    alphabet = (
        "abcdefghijklmnopqrstuvwxyz Đđăâêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ"
        "  ,.;:!?…\n0123456789%$#@-'\"() 😀"
    )
    rng = random.Random(1234)
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        duration = rng.choice([1, 2, 10, 250, 1234, 9000])
        language = rng.choice(["vi", "en"])
        assert_valid_timeline(vm.text_to_visemes(text, language, duration), duration)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Xin chào. Bạn cần gì?", ["Xin chào.", "Bạn cần gì?"]),
        ("Giá là 450.000đ rất rẻ.", ["Giá là 450.000đ rất rẻ."]),
        ("One! Two? Three; four", ["One!", "Two?", "Three;", "four"]),
        ("Dòng một\nDòng hai", ["Dòng một", "Dòng hai"]),
        ("   ", []),
        ("", []),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert speech.split_sentences(text) == expected


def test_long_sentences_are_split_without_losing_words() -> None:
    sentence = " ".join(f"word{i}" for i in range(80)) + "."
    pieces = speech.split_sentences(sentence, max_chars=60)
    assert all(0 < len(p) <= 60 for p in pieces)
    assert " ".join(pieces).split() == sentence.split()


def test_unbroken_text_is_cut_at_the_limit() -> None:
    pieces = speech.split_sentences("a" * 250, max_chars=100)
    assert [len(p) for p in pieces] == [100, 100, 50]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Xin chào bạn", "vi"),
        ("Tôi muốn mua áo", "vi"),
        ("Hello there", "en"),
        ("", "en"),
    ],
)
def test_detect_language(text: str, language: str) -> None:
    assert speech.detect_language(text) == language


# ---------------------------------------------------------------------------
# TTS through the service
# ---------------------------------------------------------------------------


async def test_every_tts_chunk_has_a_valid_viseme_timeline() -> None:
    provider = MockSpeechProvider()
    service = SpeechService(provider)
    chunks = await collect(
        service.speak("Xin chào quý khách. Bạn đang tìm áo màu gì? Tôi sẽ giúp bạn.")
    )

    assert [ch.text for ch in chunks] == [
        "Xin chào quý khách.",
        "Bạn đang tìm áo màu gì?",
        "Tôi sẽ giúp bạn.",
    ]
    for chunk in chunks:
        assert chunk.audio and chunk.audio_format == "mp3"
        assert chunk.duration_ms > 0
        assert chunk.used_fallback is True
        assert_valid_timeline(chunk.visemes, chunk.duration_ms)
    assert {lang for _, lang in provider.tts_requests} == {"vi"}


async def test_english_text_is_spoken_in_english() -> None:
    provider = MockSpeechProvider()
    chunks = await collect(SpeechService(provider).speak("Here are some shirts."))
    assert len(chunks) == 1
    assert_valid_timeline(chunks[0].visemes, chunks[0].duration_ms)
    assert provider.tts_requests == [("Here are some shirts.", "en")]


async def test_explicit_language_overrides_detection() -> None:
    provider = MockSpeechProvider()
    await collect(SpeechService(provider).speak("Hello", language="vi"))
    assert provider.tts_requests == [("Hello", "vi")]


async def test_native_visemes_are_used_and_cleaned_up() -> None:
    # Azure ids with an out-of-order event and one past the end of the audio.
    native = [(0, 0), (400, 2), (200, 21), (99_999, 15)]
    provider = MockSpeechProvider(native_visemes=native)
    (chunk,) = await collect(SpeechService(provider).speak("Xin chào"))

    assert chunk.used_fallback is False
    assert_valid_timeline(chunk.visemes, chunk.duration_ms)
    assert [e.viseme for e in chunk.visemes][:3] == ["sil", "PP", "aa"]
    assert chunk.visemes[-1].time_ms <= chunk.duration_ms


async def test_empty_text_produces_no_chunks_and_no_provider_calls() -> None:
    provider = MockSpeechProvider()
    assert await collect(SpeechService(provider).speak("  \n ")) == []
    assert provider.tts_requests == []


async def test_provider_errors_reach_the_caller() -> None:
    error = SpeechError("RATE_LIMITED", "slow down")
    service = SpeechService(MockSpeechProvider(tts_error=error))
    with pytest.raises(SpeechError, match="slow down"):
        await collect(service.speak("Hello there"))


async def test_chunk_converts_to_contract_viseme_frames() -> None:
    (chunk,) = await collect(SpeechService(MockSpeechProvider()).speak("Hello"))
    frames = chunk.viseme_frames()
    assert [(f.time_ms, f.viseme) for f in frames] == [
        (e.time_ms, e.viseme) for e in chunk.visemes
    ]


async def test_cancelling_tts_stops_new_chunks_within_100_ms() -> None:
    provider = MockSpeechProvider(tts_delay_s=0.3)
    service = SpeechService(provider)
    received: list[Any] = []
    first_chunk = asyncio.Event()

    async def consume() -> None:
        async for chunk in service.speak("Một. Hai. Ba. Bốn."):
            received.append(chunk)
            first_chunk.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first_chunk.wait(), timeout=2)

    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.perf_counter() - started

    assert elapsed < CANCEL_BUDGET_S
    assert len(received) == 1
    await asyncio.sleep(0.5)  # no further synthesis starts after the cancel
    assert len(provider.tts_requests) == 2  # the 2nd was in flight when cancelled
    assert provider.tts_closed == 2  # and every started stream was closed


async def test_stopping_early_closes_the_provider_stream() -> None:
    provider = MockSpeechProvider()
    stream = SpeechService(provider).speak("Một. Hai. Ba.")
    async for _ in stream:
        break
    await stream.aclose()
    assert len(provider.tts_requests) == 1
    assert provider.tts_closed == 1


# ---------------------------------------------------------------------------
# STT through the service
# ---------------------------------------------------------------------------


def test_aggregator_joins_finished_segments() -> None:
    agg = SttAggregator("vi")
    steps = [
        SttSegment("tìm", 0.0, "vi", False),
        SttSegment("tìm áo", 0.0, "vi", False),
        SttSegment("tìm áo giống", 0.90, "vi", True),
        SttSegment("màu", 0.0, "vi", False),
        SttSegment("màu này", 0.80, "vi", True),
    ]
    running = [agg.add(s) for s in steps]

    assert [r.text for r in running] == [
        "tìm",
        "tìm áo",
        "tìm áo giống",
        "tìm áo giống màu",
        "tìm áo giống màu này",
    ]
    assert all(r.is_final is False for r in running)
    final = agg.final()
    assert final.is_final is True
    assert final.text == "tìm áo giống màu này"
    expected = (len("tìm áo giống") * 0.90 + len("màu này") * 0.80) / (
        len("tìm áo giống") + len("màu này")
    )
    assert final.confidence == pytest.approx(expected)


def test_aggregator_follows_the_detected_language() -> None:
    agg = SttAggregator("vi")
    agg.add(SttSegment("hello", 0.9, "en", True))
    assert agg.final().language == "en"


def test_aggregator_with_nothing_recognized() -> None:
    final = SttAggregator("vi").final()
    assert (final.text, final.confidence, final.is_final) == ("", 0.0, True)


async def test_transcribe_streams_partials_then_one_final() -> None:
    provider = MockSpeechProvider(transcript="Tìm áo giống màu này", confidence=0.93)
    service = SpeechService(provider)
    results = await collect(service.transcribe(pcm_frames(5), language="vi"))

    assert sum(r.is_final for r in results) == 1
    assert results[-1].is_final is True
    assert results[-1].text == "Tìm áo giống màu này"
    assert results[-1].confidence == pytest.approx(0.93)
    assert results[-1].language == "vi"
    assert results[0].is_final is False

    session = provider.sessions[0]
    assert len(session.frames) == 5
    assert session.closed is True


async def test_transcribe_without_audio_returns_an_empty_final() -> None:
    service = SpeechService(MockSpeechProvider())
    results = await collect(service.transcribe(pcm_frames(0)))
    assert [(r.text, r.confidence, r.is_final) for r in results] == [("", 0.0, True)]


async def test_transcribe_skips_empty_frames() -> None:
    async def frames() -> AsyncIterator[bytes]:
        yield b""
        yield b"\x01\x00" * 10
        yield b""

    provider = MockSpeechProvider()
    await collect(SpeechService(provider).transcribe(frames()))
    assert len(provider.sessions[0].frames) == 1


async def test_transcribe_rejects_half_samples() -> None:
    async def frames() -> AsyncIterator[bytes]:
        yield b"\x01\x00\x02"

    with pytest.raises(ValueError, match="16-bit"):
        await collect(SpeechService(MockSpeechProvider()).transcribe(frames()))


async def test_stt_errors_reach_the_caller_and_close_the_session() -> None:
    provider = MockSpeechProvider(stt_error=SpeechError("STT_FAILED", "no recogniser"))
    with pytest.raises(SpeechError, match="no recogniser"):
        await collect(SpeechService(provider).transcribe(pcm_frames(3)))
    assert provider.sessions[0].closed is True


async def test_cancelling_stt_closes_the_session_within_100_ms() -> None:
    gate = asyncio.Event()

    async def endless_frames() -> AsyncIterator[bytes]:
        while not gate.is_set():
            yield b"\x01\x00" * 320
            await asyncio.sleep(0.02)

    provider = MockSpeechProvider()
    service = SpeechService(provider)
    task = asyncio.create_task(collect(service.transcribe(endless_frames())))
    await asyncio.sleep(0.1)

    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.perf_counter() - started
    gate.set()

    assert elapsed < CANCEL_BUDGET_S
    assert provider.sessions[0].closed is True


# ---------------------------------------------------------------------------
# Azure adapter, against a fake Speech SDK
# ---------------------------------------------------------------------------


class FakeSignal:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def connect(self, handler: Any) -> None:
        self.handlers.append(handler)

    def fire(self, *args: Any) -> None:
        for handler in list(self.handlers):
            handler(*args)


def make_fake_sdk(
    *,
    tts: dict[str, Any] | None = None,
    stt_script: list[tuple[Any, ...]] | None = None,
) -> tuple[Any, SimpleNamespace]:
    """A stand-in for azure.cognitiveservices.speech with scripted behaviour."""
    tts_cfg: dict[str, Any] = {
        "events": [],
        "audio": b"\x00" * 6000,
        "duration": None,
        "ok": True,
        "error": "",
        "block": None,
        **(tts or {}),
    }
    record = SimpleNamespace(
        configs=[], synthesizers=[], recognizers=[], in_get=threading.Event()
    )

    class ResultReason:
        RecognizedSpeech = "RecognizedSpeech"
        NoMatch = "NoMatch"
        SynthesizingAudioCompleted = "SynthesizingAudioCompleted"
        Canceled = "Canceled"

    class CancellationReason:
        Error = "Error"
        EndOfStream = "EndOfStream"

    class OutputFormat:
        Detailed = "Detailed"

    class PropertyId:
        SpeechServiceResponse_JsonResult = "json"

    class SpeechSynthesisOutputFormat:
        Audio24Khz48KBitRateMonoMp3 = "Audio24Khz48KBitRateMonoMp3"

    class SpeechConfig:
        def __init__(self, subscription: str, region: str) -> None:
            self.subscription, self.region = subscription, region
            self.output_format_set: Any = None
            record.configs.append(self)

        def set_speech_synthesis_output_format(self, fmt: Any) -> None:
            self.output_format_set = fmt

    class Pending:
        def __init__(self, fn: Any) -> None:
            self._fn = fn

        def get(self) -> Any:
            return self._fn()

    class SpeechSynthesizer:
        def __init__(self, speech_config: Any, audio_config: Any) -> None:
            self.config, self.audio_config = speech_config, audio_config
            self.viseme_received = FakeSignal()
            self.stop_calls = 0
            record.synthesizers.append(self)

        def speak_text_async(self, text: str) -> Pending:
            def run() -> Any:
                record.in_get.set()
                if tts_cfg["block"] is not None:
                    tts_cfg["block"].wait(timeout=5)
                for ticks, viseme_id in tts_cfg["events"]:
                    self.viseme_received.fire(
                        SimpleNamespace(audio_offset=ticks, viseme_id=viseme_id)
                    )
                result = SimpleNamespace(
                    reason=(
                        ResultReason.SynthesizingAudioCompleted
                        if tts_cfg["ok"]
                        else ResultReason.Canceled
                    ),
                    audio_data=tts_cfg["audio"],
                    cancellation_details=SimpleNamespace(
                        error_details=tts_cfg["error"]
                    ),
                )
                if tts_cfg["duration"] is not None:
                    result.audio_duration = tts_cfg["duration"]
                return result

            return Pending(run)

        def stop_speaking_async(self) -> Pending:
            self.stop_calls += 1
            return Pending(lambda: None)

    class PushAudioInputStream:
        def __init__(self, stream_format: Any = None) -> None:
            self.written: list[bytes] = []
            self.closed = False
            self.on_close: Any = None

        def write(self, data: bytes) -> None:
            self.written.append(data)

        def close(self) -> None:
            if not self.closed:
                self.closed = True
                if self.on_close:
                    self.on_close()

    class Props:
        def __init__(self, json_result: str | None) -> None:
            self._json = json_result

        def get(self, key: Any) -> str | None:
            return self._json

    class SpeechRecognizer:
        def __init__(
            self,
            speech_config: Any,
            audio_config: Any,
            auto_detect_source_language_config: Any = None,
        ) -> None:
            self.config = speech_config
            self.auto_detect = auto_detect_source_language_config
            self.recognizing, self.recognized = FakeSignal(), FakeSignal()
            self.canceled, self.session_stopped = FakeSignal(), FakeSignal()
            self.stopped = False
            audio_config.stream.on_close = self._audio_ended
            record.recognizers.append(self)

        def start_continuous_recognition_async(self) -> Pending:
            return Pending(lambda: None)

        def stop_continuous_recognition_async(self) -> Pending:
            self.stopped = True
            return Pending(lambda: None)

        def _audio_ended(self) -> None:
            threading.Thread(target=self._play_script, daemon=True).start()

        def _result(
            self, text: str, reason: str, conf: float | None, locale: str
        ) -> Any:
            payload = (
                None if conf is None else json.dumps({"NBest": [{"Confidence": conf}]})
            )
            return SimpleNamespace(
                text=text, reason=reason, properties=Props(payload), language=locale
            )

        def _play_script(self) -> None:
            for step in stt_script or []:
                kind = step[0]
                if kind == "partial":
                    result = self._result(
                        step[1], ResultReason.RecognizedSpeech, None, step[2]
                    )
                    self.recognizing.fire(SimpleNamespace(result=result))
                elif kind == "final":
                    result = self._result(
                        step[1], ResultReason.RecognizedSpeech, step[2], step[3]
                    )
                    self.recognized.fire(SimpleNamespace(result=result))
                elif kind == "nomatch":
                    result = self._result("", ResultReason.NoMatch, None, "vi-VN")
                    self.recognized.fire(SimpleNamespace(result=result))
                elif kind == "error":
                    details = SimpleNamespace(
                        reason=CancellationReason.Error, error_details=step[1]
                    )
                    self.canceled.fire(SimpleNamespace(cancellation_details=details))
                    return
            self.session_stopped.fire(SimpleNamespace())

    sdk = SimpleNamespace(
        ResultReason=ResultReason,
        CancellationReason=CancellationReason,
        OutputFormat=OutputFormat,
        PropertyId=PropertyId,
        SpeechSynthesisOutputFormat=SpeechSynthesisOutputFormat,
        SpeechConfig=SpeechConfig,
        SpeechSynthesizer=SpeechSynthesizer,
        SpeechRecognizer=SpeechRecognizer,
        AutoDetectSourceLanguageResult=lambda r: SimpleNamespace(language=r.language),
        audio=SimpleNamespace(
            AudioStreamFormat=lambda **kw: SimpleNamespace(**kw),
            PushAudioInputStream=PushAudioInputStream,
            AudioConfig=lambda stream: SimpleNamespace(stream=stream),
        ),
        languageconfig=SimpleNamespace(
            AutoDetectSourceLanguageConfig=lambda languages: SimpleNamespace(
                languages=languages
            )
        ),
    )
    return sdk, record


def azure_provider(sdk: Any) -> AzureSpeechProvider:
    return AzureSpeechProvider("key", "southeastasia", sdk=sdk)


def test_azure_helper_functions() -> None:
    assert speech.azure_locale("vi") == "vi-VN"
    assert speech.azure_locale("en") == "en-US"
    assert speech.language_from_locale("vi-VN") == "vi"
    assert speech.language_from_locale("en-US") == "en"
    assert speech.language_from_locale("fr-FR") is None
    assert speech.language_from_locale(None) is None
    assert speech.ticks_to_ms(10_000_000) == 1000  # 100 ns ticks
    assert speech.ticks_to_ms(1_234_567) == 123


def test_parse_confidence() -> None:
    good = json.dumps({"NBest": [{"Confidence": 0.87}]})
    assert speech.parse_confidence(good) == pytest.approx(0.87)
    assert speech.parse_confidence(json.dumps({"NBest": [{"Confidence": 7}]})) == 1.0
    for bad in (None, "", "not json", json.dumps({}), json.dumps({"NBest": []})):
        assert speech.parse_confidence(bad) == 0.0


async def test_azure_tts_converts_native_visemes() -> None:
    # Offsets of 0, 120, 300 and 450 ms, in 100 ns ticks.
    ticks = [(0, 0), (1_200_000, 2), (3_000_000, 21), (4_500_000, 0)]
    sdk, record = make_fake_sdk(
        tts={"events": ticks, "duration": timedelta(milliseconds=520)}
    )
    service = SpeechService(azure_provider(sdk))
    (chunk,) = await collect(service.speak("Xin chào"))

    assert chunk.used_fallback is False
    assert chunk.duration_ms == 520
    assert chunk.audio_format == "mp3"
    assert [(e.time_ms, e.viseme) for e in chunk.visemes] == [
        (0, "sil"),
        (120, "aa"),
        (300, "PP"),
        (450, "sil"),
    ]
    assert_valid_timeline(chunk.visemes, chunk.duration_ms)

    config = record.configs[0]
    assert config.speech_synthesis_voice_name == "vi-VN-HoaiMyNeural"
    assert config.output_format_set == "Audio24Khz48KBitRateMonoMp3"
    assert record.synthesizers[0].audio_config is None  # audio stays in memory


async def test_azure_tts_picks_the_english_voice() -> None:
    sdk, record = make_fake_sdk()
    await collect(SpeechService(azure_provider(sdk)).speak("Hello there"))
    assert record.configs[0].speech_synthesis_voice_name == "en-US-JennyNeural"


async def test_azure_tts_duration_falls_back_to_the_byte_count() -> None:
    sdk, _ = make_fake_sdk(tts={"audio": b"\x00" * 6000, "duration": None})
    (chunk,) = await collect(SpeechService(azure_provider(sdk)).speak("Hello"))
    assert chunk.duration_ms == 1000  # 48 kbps = 6000 bytes per second


async def test_azure_tts_without_viseme_events_uses_the_fallback() -> None:
    sdk, _ = make_fake_sdk(tts={"events": [], "duration": timedelta(milliseconds=800)})
    (chunk,) = await collect(SpeechService(azure_provider(sdk)).speak("Xin chào"))
    assert chunk.used_fallback is True
    assert_valid_timeline(chunk.visemes, chunk.duration_ms)


async def test_azure_tts_failure_is_reported() -> None:
    sdk, _ = make_fake_sdk(tts={"ok": False, "error": "Connection failed"})
    with pytest.raises(SpeechError) as caught:
        await collect(SpeechService(azure_provider(sdk)).speak("Hello"))
    assert caught.value.code == "INTERNAL"


async def test_azure_tts_throttling_maps_to_rate_limited() -> None:
    sdk, _ = make_fake_sdk(tts={"ok": False, "error": "HTTP 429 Too Many Requests"})
    with pytest.raises(SpeechError) as caught:
        await collect(SpeechService(azure_provider(sdk)).speak("Hello"))
    assert caught.value.code == "RATE_LIMITED"


async def test_azure_tts_cancel_stops_speaking_within_100_ms() -> None:
    block = threading.Event()
    sdk, record = make_fake_sdk(tts={"block": block})
    service = SpeechService(azure_provider(sdk))
    task = asyncio.create_task(collect(service.speak("Hello there")))

    deadline = time.perf_counter() + 2
    while not record.in_get.is_set() and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    assert record.in_get.is_set()  # the blocking SDK call is now in flight

    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.perf_counter() - started
    block.set()  # let the worker thread finish

    assert elapsed < CANCEL_BUDGET_S
    assert record.synthesizers[0].stop_calls == 1


async def test_azure_stt_streams_partials_and_joins_segments() -> None:
    script = [
        ("partial", "tìm", "vi-VN"),
        ("final", "tìm áo giống", 0.90, "vi-VN"),
        ("nomatch",),
        ("partial", "màu", "vi-VN"),
        ("final", "màu này", 0.80, "vi-VN"),
    ]
    sdk, record = make_fake_sdk(stt_script=script)
    service = SpeechService(azure_provider(sdk))
    results = await collect(service.transcribe(pcm_frames(4)))

    assert results[-1].is_final is True
    assert results[-1].text == "tìm áo giống màu này"
    assert results[-1].language == "vi"
    assert 0.80 < results[-1].confidence < 0.90
    assert sum(r.is_final for r in results) == 1
    assert [r.text for r in results[:-1]][0] == "tìm"

    recognizer = record.recognizers[0]
    assert recognizer.stopped is True  # closed after use
    assert recognizer.config.output_format == "Detailed"


async def test_azure_stt_detects_between_vietnamese_and_english() -> None:
    sdk, record = make_fake_sdk(stt_script=[("final", "show me shirts", 0.95, "en-US")])
    service = SpeechService(azure_provider(sdk))
    results = await collect(service.transcribe(pcm_frames(2)))

    assert results[-1].language == "en"
    assert record.recognizers[0].auto_detect.languages == ["vi-VN", "en-US"]


async def test_azure_stt_uses_a_fixed_language_when_given() -> None:
    sdk, record = make_fake_sdk(stt_script=[("final", "xin chào", 0.9, "vi-VN")])
    await collect(SpeechService(azure_provider(sdk)).transcribe(pcm_frames(1), "vi"))

    recognizer = record.recognizers[0]
    assert recognizer.auto_detect is None
    assert recognizer.config.speech_recognition_language == "vi-VN"


async def test_azure_stt_error_is_reported() -> None:
    sdk, _ = make_fake_sdk(stt_script=[("error", "Authentication failed")])
    with pytest.raises(SpeechError) as caught:
        await collect(SpeechService(azure_provider(sdk)).transcribe(pcm_frames(2)))
    assert caught.value.code == "STT_FAILED"


async def test_azure_stt_with_no_speech_gives_an_empty_final() -> None:
    sdk, _ = make_fake_sdk(stt_script=[("nomatch",)])
    service = SpeechService(azure_provider(sdk))
    results = await collect(service.transcribe(pcm_frames(2)))
    assert [(r.text, r.confidence, r.is_final) for r in results] == [("", 0.0, True)]


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_azure_provider_needs_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_SPEECH_KEY"):
        AzureSpeechProvider.from_env()


def test_factory_defaults_to_mock_and_rejects_unknown_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speech.get_speech_service.cache_clear()
    monkeypatch.delenv("SPEECH_PROVIDER", raising=False)
    assert speech.get_speech_service().provider_name == "mock"

    speech.get_speech_service.cache_clear()
    monkeypatch.setenv("SPEECH_PROVIDER", "carrier-pigeon")
    with pytest.raises(ValueError, match="carrier-pigeon"):
        speech.get_speech_service()
    speech.get_speech_service.cache_clear()
