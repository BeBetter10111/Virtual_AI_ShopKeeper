"""Speech pipeline: streaming speech-to-text and chunked text-to-speech with visemes.

Backends sit behind the `SpeechProvider` interface. Two are included:
    mock   no cloud account needed (scripted STT, fake audio); the default
    azure  Azure AI Speech: vi-VN and en-US STT/TTS with native viseme events

Environment variables:
    SPEECH_PROVIDER      "mock" (default) or "azure"
    AZURE_SPEECH_KEY     Azure Speech resource key
    AZURE_SPEECH_REGION  e.g. "southeastasia"
    AZURE_VOICE_VI       default "vi-VN-HoaiMyNeural"
    AZURE_VOICE_EN       default "en-US-JennyNeural"

Cancellation: `SpeechService.speak` and `SpeechService.transcribe` are async
generators. Cancelling the task that consumes them (barge-in) stops them at the next
await and releases provider resources in `finally` blocks.
"""

import asyncio
import contextlib
import functools
import json
import logging
import os
import re
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.utils.viseme_mapper import (
    VisemeEvent,
    from_azure_events,
    sanitize_visemes,
    text_to_visemes,
)

logger = logging.getLogger("app.speech")

Language = Literal["vi", "en"]
AudioFormat = Literal["mp3", "opus"]
SpeechErrorCode = Literal["STT_FAILED", "RATE_LIMITED", "INTERNAL"]

PCM_FRAME_BYTES = 640  # 20 ms of 16 kHz, 16-bit, mono audio
MAX_CHUNK_CHARS = 140
MP3_48KBPS_BYTES_PER_SECOND = 6000  # constant-bitrate output used with Azure

_VIETNAMESE_CHARS = frozenset(
    "ăâđêôơư"
    "áàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)


class SpeechError(Exception):
    """A speech backend failure, mapped to the WebSocket ERROR codes."""

    def __init__(
        self, code: SpeechErrorCode, message: str, recoverable: bool = True
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SttSegment:
    """One recognition event from a provider (a partial or a finished segment)."""

    text: str
    confidence: float
    language: Language | None
    is_final: bool


@dataclass(frozen=True)
class SttResult:
    text: str
    confidence: float
    is_final: bool
    language: Language


@dataclass(frozen=True)
class ProviderChunk:
    """Audio for one piece of text. `visemes` is None if the provider has none."""

    text: str
    audio: bytes
    audio_format: AudioFormat
    duration_ms: int
    visemes: list[VisemeEvent] | None


@dataclass(frozen=True)
class SpeechChunk:
    """A TTS chunk ready to send: audio plus a valid viseme timeline."""

    text: str
    audio: bytes
    audio_format: AudioFormat
    duration_ms: int
    visemes: list[VisemeEvent]
    used_fallback: bool  # visemes were derived from text, not from the provider

    def viseme_frames(self) -> list[Any]:
        """The timeline as `contracts.VisemeFrame` models (for WebSocket messages)."""
        from app.schemas import contracts as c

        return [c.VisemeFrame(time_ms=v.time_ms, viseme=v.viseme) for v in self.visemes]


class SttSession(Protocol):
    async def push_audio(self, pcm: bytes) -> None: ...

    async def finish(self) -> None:
        """No more audio will be sent."""
        ...

    def segments(self) -> AsyncGenerator[SttSegment, None]:
        """Recognition events; ends after `finish()` once everything is recognized."""
        ...

    async def close(self) -> None:
        """Abort and release resources (safe to call more than once)."""
        ...


class SpeechProvider(Protocol):
    name: str

    async def open_stt(self, language: Language | None) -> SttSession: ...

    def synthesize(
        self, text: str, language: Language
    ) -> AsyncGenerator[ProviderChunk, None]:
        """Speak `text`; yields chunks. Must stop promptly when cancelled."""
        ...


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def detect_language(text: str) -> Language:
    """"vi" if the text contains Vietnamese-specific letters, otherwise "en"."""
    return "vi" if any(ch in _VIETNAMESE_CHARS for ch in text.lower()) else "en"


_SENTENCE_END = re.compile(r"(?<=[.!?…;])\s+|\n+")


def split_sentences(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into speakable pieces, each at most `max_chars` long.

    Splits after . ! ? … ; when followed by whitespace (so "450.000đ" stays whole),
    and at line breaks; over-long sentences are broken at commas, then at spaces.
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = max(
                sentence.rfind(", ", 0, max_chars), sentence.rfind(" ", 0, max_chars)
            )
            if cut <= 0:
                cut = max_chars
            pieces.append(sentence[:cut].strip(" ,"))
            sentence = sentence[cut:].strip(" ,")
        if sentence:
            pieces.append(sentence)
    return [p for p in pieces if p]


# ---------------------------------------------------------------------------
# STT aggregation
# ---------------------------------------------------------------------------


class SttAggregator:
    """Combine provider segments into running transcripts and one final result.

    Continuous recognition may split one utterance into several finished segments
    (after short pauses); they are joined into a single transcript.
    """

    def __init__(self, default_language: Language) -> None:
        self._language: Language = default_language
        self._finished: list[tuple[str, float]] = []

    def _confidence(self) -> float:
        total = sum(len(text) for text, _ in self._finished)
        if total == 0:
            return 0.0
        return sum(len(text) * conf for text, conf in self._finished) / total

    def add(self, segment: SttSegment) -> SttResult:
        """Fold in a segment; returns the current (not final) transcript."""
        if segment.language is not None:
            self._language = segment.language
        text = segment.text.strip()
        if segment.is_final:
            if text:
                self._finished.append((text, segment.confidence))
            current = [t for t, _ in self._finished]
        else:
            current = [t for t, _ in self._finished] + ([text] if text else [])
        confidence = self._confidence() if self._finished else segment.confidence
        return SttResult(" ".join(current), confidence, False, self._language)

    def final(self) -> SttResult:
        text = " ".join(t for t, _ in self._finished)
        return SttResult(text, self._confidence(), True, self._language)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SpeechService:
    def __init__(
        self, provider: SpeechProvider, max_chunk_chars: int = MAX_CHUNK_CHARS
    ) -> None:
        self._provider = provider
        self._max_chunk_chars = max_chunk_chars

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def transcribe(
        self,
        frames: AsyncIterator[bytes],
        language: Language | None = None,
    ) -> AsyncIterator[SttResult]:
        """Recognize speech from 16 kHz / 16-bit / mono PCM frames.

        Yields running transcripts (`is_final=False`) and, when `frames` ends, exactly
        one final transcript (empty text and confidence 0 if nothing was recognized).
        """
        session = await self._provider.open_stt(language)
        aggregator = SttAggregator(language or "vi")
        pump = asyncio.create_task(self._pump(frames, session))
        try:
            async with contextlib.aclosing(session.segments()) as segments:
                async for segment in segments:
                    yield aggregator.add(segment)
            await pump  # surfaces errors from the audio source
            yield aggregator.final()
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
            with contextlib.suppress(Exception):
                await session.close()

    @staticmethod
    async def _pump(frames: AsyncIterator[bytes], session: SttSession) -> None:
        try:
            async for frame in frames:
                if not frame:
                    continue
                if len(frame) % 2:
                    raise ValueError("PCM frames must contain whole 16-bit samples.")
                await session.push_audio(frame)
        finally:
            await session.finish()

    async def speak(
        self, text: str, language: Language | None = None
    ) -> AsyncIterator[SpeechChunk]:
        """Synthesize `text` sentence by sentence.

        Every yielded chunk has a valid viseme timeline: the provider's own when it has
        one, otherwise one derived from the text. The first chunk is produced after
        synthesizing only the first sentence, which keeps time-to-first-audio low.
        """
        spoken_language = language or detect_language(text)
        for piece in split_sentences(text, self._max_chunk_chars):
            stream = self._provider.synthesize(piece, spoken_language)
            async with contextlib.aclosing(stream) as chunks:
                async for chunk in chunks:
                    yield self._finalize(chunk, spoken_language)

    @staticmethod
    def _finalize(chunk: ProviderChunk, language: Language) -> SpeechChunk:
        used_fallback = chunk.visemes is None
        raw = (
            text_to_visemes(chunk.text, language, chunk.duration_ms)
            if chunk.visemes is None
            else chunk.visemes
        )
        return SpeechChunk(
            text=chunk.text,
            audio=chunk.audio,
            audio_format=chunk.audio_format,
            duration_ms=chunk.duration_ms,
            visemes=sanitize_visemes(raw, chunk.duration_ms),
            used_fallback=used_fallback,
        )


# ---------------------------------------------------------------------------
# Mock provider (development and tests)
# ---------------------------------------------------------------------------


class MockSttSession:
    def __init__(
        self,
        transcript: str,
        language: Language,
        confidence: float,
        error: SpeechError | None,
    ) -> None:
        self._transcript = transcript
        self._language: Language = language
        self._confidence = confidence
        self._error = error
        self._finished = asyncio.Event()
        self.frames: list[bytes] = []
        self.closed = False

    async def push_audio(self, pcm: bytes) -> None:
        self.frames.append(pcm)

    async def finish(self) -> None:
        self._finished.set()

    async def segments(self) -> AsyncGenerator[SttSegment, None]:
        await self._finished.wait()
        if self._error is not None:
            raise self._error
        if not self.frames:
            return  # no audio: nothing recognized
        words = self._transcript.split()
        if len(words) > 1:
            partial = " ".join(words[: len(words) // 2])
            yield SttSegment(partial, 0.0, self._language, False)
        yield SttSegment(self._transcript, self._confidence, self._language, True)

    async def close(self) -> None:
        self.closed = True
        self._finished.set()


class MockSpeechProvider:
    """Scripted provider that needs no network.

    STT returns `transcript` once audio has been sent. TTS returns placeholder audio
    bytes (not playable) with a duration proportional to the text length; visemes are
    left to the text fallback unless `native_visemes` is given as (offset_ms, azure_id)
    pairs.
    """

    name = "mock"

    def __init__(
        self,
        transcript: str = "Tìm áo giống màu này",
        language: Language = "vi",
        confidence: float = 0.93,
        tts_delay_s: float = 0.0,
        native_visemes: list[tuple[int, int]] | None = None,
        stt_error: SpeechError | None = None,
        tts_error: SpeechError | None = None,
    ) -> None:
        self.transcript = transcript
        self.language: Language = language
        self.confidence = confidence
        self.tts_delay_s = tts_delay_s
        self.native_visemes = native_visemes
        self.stt_error = stt_error
        self.tts_error = tts_error
        self.sessions: list[MockSttSession] = []
        self.tts_requests: list[tuple[str, Language]] = []
        self.tts_closed = 0

    async def open_stt(self, language: Language | None) -> SttSession:
        session = MockSttSession(
            self.transcript, language or self.language, self.confidence, self.stt_error
        )
        self.sessions.append(session)
        return session

    async def synthesize(
        self, text: str, language: Language
    ) -> AsyncGenerator[ProviderChunk, None]:
        self.tts_requests.append((text, language))
        try:
            if self.tts_delay_s:
                await asyncio.sleep(self.tts_delay_s)
            if self.tts_error is not None:
                raise self.tts_error
            visemes = (
                from_azure_events(self.native_visemes)
                if self.native_visemes is not None
                else None
            )
            yield ProviderChunk(
                text=text,
                audio=b"MOCK-MP3:" + text.encode("utf-8"),
                audio_format="mp3",
                duration_ms=max(250, 55 * len(text)),
                visemes=visemes,
            )
        finally:
            self.tts_closed += 1


# ---------------------------------------------------------------------------
# Azure AI Speech provider
# ---------------------------------------------------------------------------

_AZURE_LOCALES: dict[Language, str] = {"vi": "vi-VN", "en": "en-US"}


def azure_locale(language: Language) -> str:
    return _AZURE_LOCALES[language]


def language_from_locale(locale: str | None) -> Language | None:
    if not locale:
        return None
    prefix = locale.split("-")[0].lower()
    return "vi" if prefix == "vi" else "en" if prefix == "en" else None


def ticks_to_ms(ticks: int) -> int:
    """Azure audio offsets are in 100-nanosecond ticks."""
    return ticks // 10_000


def parse_confidence(json_result: str | None) -> float:
    """Confidence of the best hypothesis from Azure's detailed JSON result."""
    if not json_result:
        return 0.0
    try:
        best = json.loads(json_result)["NBest"][0]
        return min(1.0, max(0.0, float(best["Confidence"])))
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def _azure_error(details: str, default: SpeechErrorCode) -> SpeechError:
    lowered = details.lower()
    if "429" in lowered or "too many requests" in lowered:
        return SpeechError("RATE_LIMITED", "Speech service is rate limited.")
    return SpeechError(default, f"Speech service error: {details[:200]}")


_END = object()


class AzureSttSession:
    """Continuous recognition on a push stream. SDK callbacks run on SDK threads."""

    def __init__(
        self, sdk: Any, key: str, region: str, language: Language | None
    ) -> None:
        self._sdk = sdk
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._closed = False

        config = sdk.SpeechConfig(subscription=key, region=region)
        config.output_format = sdk.OutputFormat.Detailed  # includes confidence
        stream_format = sdk.audio.AudioStreamFormat(
            samples_per_second=16000, bits_per_sample=16, channels=1
        )
        self._push = sdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio = sdk.audio.AudioConfig(stream=self._push)

        if language is None:
            auto = sdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=list(_AZURE_LOCALES.values())
            )
            self._recognizer = sdk.SpeechRecognizer(
                speech_config=config,
                audio_config=audio,
                auto_detect_source_language_config=auto,
            )
        else:
            config.speech_recognition_language = azure_locale(language)
            self._recognizer = sdk.SpeechRecognizer(
                speech_config=config, audio_config=audio
            )
        self._recognizer.recognizing.connect(lambda e: self._on_result(e, False))
        self._recognizer.recognized.connect(lambda e: self._on_result(e, True))
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.session_stopped.connect(lambda e: self._emit(_END))

    async def start(self) -> None:
        await asyncio.to_thread(
            lambda: self._recognizer.start_continuous_recognition_async().get()
        )

    def _emit(self, item: object) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def _on_result(self, event: Any, is_final: bool) -> None:
        sdk, result = self._sdk, event.result
        if is_final and result.reason != sdk.ResultReason.RecognizedSpeech:
            return  # NoMatch
        locale = None
        with contextlib.suppress(Exception):
            locale = sdk.AutoDetectSourceLanguageResult(result).language
        confidence = (
            parse_confidence(
                result.properties.get(sdk.PropertyId.SpeechServiceResponse_JsonResult)
            )
            if is_final
            else 0.0
        )
        self._emit(
            SttSegment(result.text, confidence, language_from_locale(locale), is_final)
        )

    def _on_canceled(self, event: Any) -> None:
        sdk = self._sdk
        details = event.cancellation_details
        if details.reason == sdk.CancellationReason.Error:
            self._emit(_azure_error(str(details.error_details), "STT_FAILED"))
        self._emit(_END)

    async def push_audio(self, pcm: bytes) -> None:
        self._push.write(pcm)

    async def finish(self) -> None:
        self._push.close()

    async def segments(self) -> AsyncGenerator[SttSegment, None]:
        while True:
            item = await self._queue.get()
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, SttSegment)
            yield item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._push.close()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                lambda: self._recognizer.stop_continuous_recognition_async().get()
            )


class AzureSpeechProvider:
    """Azure AI Speech (Speech SDK). Needs `azure-cognitiveservices-speech`.

    TTS uses constant-bitrate 48 kbps MP3 and returns one chunk per call, with the
    native viseme events converted to Oculus names and millisecond offsets. A fresh
    synthesizer per call keeps calls independent; reusing one with a pre-opened
    connection is a possible latency optimisation (measure first).
    """

    name = "azure"

    def __init__(
        self,
        key: str,
        region: str,
        voice_vi: str = "vi-VN-HoaiMyNeural",
        voice_en: str = "en-US-JennyNeural",
        sdk: Any = None,
    ) -> None:
        self._key = key
        self._region = region
        self._voices: dict[Language, str] = {"vi": voice_vi, "en": voice_en}
        self._sdk = sdk

    @classmethod
    def from_env(cls) -> "AzureSpeechProvider":
        key = os.environ.get("AZURE_SPEECH_KEY")
        region = os.environ.get("AZURE_SPEECH_REGION")
        if not key or not region:
            raise RuntimeError(
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set for the "
                "azure speech provider."
            )
        return cls(
            key,
            region,
            os.environ.get("AZURE_VOICE_VI", "vi-VN-HoaiMyNeural"),
            os.environ.get("AZURE_VOICE_EN", "en-US-JennyNeural"),
        )

    def _speech_sdk(self) -> Any:
        if self._sdk is None:
            try:
                import azure.cognitiveservices.speech as sdk
            except ImportError as exc:
                raise RuntimeError(
                    "Install azure-cognitiveservices-speech to use the azure provider."
                ) from exc
            self._sdk = sdk
        return self._sdk

    async def open_stt(self, language: Language | None) -> SttSession:
        session = AzureSttSession(
            self._speech_sdk(), self._key, self._region, language
        )
        try:
            await session.start()
        except Exception as exc:
            raise _azure_error(str(exc), "STT_FAILED") from exc
        return session

    def _synthesize_blocking(
        self, text: str, language: Language, holder: dict[str, Any]
    ) -> ProviderChunk:
        sdk = self._speech_sdk()
        config = sdk.SpeechConfig(subscription=self._key, region=self._region)
        config.speech_synthesis_voice_name = self._voices[language]
        config.set_speech_synthesis_output_format(
            sdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )
        synthesizer = sdk.SpeechSynthesizer(speech_config=config, audio_config=None)
        holder["synthesizer"] = synthesizer

        events: list[tuple[int, int]] = []
        synthesizer.viseme_received.connect(
            lambda e: events.append((ticks_to_ms(e.audio_offset), int(e.viseme_id)))
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason != sdk.ResultReason.SynthesizingAudioCompleted:
            details = result.cancellation_details
            raise _azure_error(str(details.error_details), "INTERNAL")

        audio = bytes(result.audio_data)
        reported = getattr(result, "audio_duration", None)
        duration_ms = (
            int(reported.total_seconds() * 1000)
            if reported is not None
            else len(audio) * 1000 // MP3_48KBPS_BYTES_PER_SECOND
        )
        return ProviderChunk(
            text=text,
            audio=audio,
            audio_format="mp3",
            duration_ms=duration_ms,
            visemes=from_azure_events(events) if events else None,
        )

    async def synthesize(
        self, text: str, language: Language
    ) -> AsyncGenerator[ProviderChunk, None]:
        holder: dict[str, Any] = {}
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, self._synthesize_blocking, text, language, holder
        )
        try:
            chunk = await future
        except asyncio.CancelledError:
            synthesizer = holder.get("synthesizer")
            if synthesizer is not None:
                with contextlib.suppress(Exception):
                    synthesizer.stop_speaking_async()
            raise
        yield chunk


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@functools.cache
def get_speech_service() -> SpeechService:
    """Process-wide service for the provider chosen by SPEECH_PROVIDER."""
    kind = os.environ.get("SPEECH_PROVIDER", "mock").strip().lower()
    provider: SpeechProvider
    if kind == "azure":
        provider = AzureSpeechProvider.from_env()
    elif kind == "mock":
        logger.warning("Using the mock speech provider (set SPEECH_PROVIDER=azure).")
        provider = MockSpeechProvider()
    else:
        raise ValueError(f"Unknown SPEECH_PROVIDER {kind!r}; use 'mock' or 'azure'.")
    return SpeechService(provider)


__all__ = [
    "AzureSpeechProvider",
    "MockSpeechProvider",
    "ProviderChunk",
    "SpeechChunk",
    "SpeechError",
    "SpeechProvider",
    "SpeechService",
    "SttAggregator",
    "SttResult",
    "SttSegment",
    "SttSession",
    "detect_language",
    "get_speech_service",
    "split_sentences",
]
