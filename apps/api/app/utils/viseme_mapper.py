"""Viseme helpers: provider IDs -> Oculus 15, a text-based fallback, and validation.

Every chunk of synthesized speech must carry a valid viseme timeline: Oculus names,
`time_ms` strictly increasing, all within the chunk's audio duration. `sanitize_visemes`
enforces that for any source, and `text_to_visemes` supplies a timeline when the speech
provider has no native visemes (approximate: it works from spelling, not phonemes).
"""

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, get_args

Viseme = Literal[
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O",
    "U",
]
OCULUS_VISEMES: tuple[Viseme, ...] = get_args(Viseme)
_OCULUS_SET = frozenset(OCULUS_VISEMES)


@dataclass(frozen=True)
class VisemeEvent:
    time_ms: int  # relative to the start of the audio chunk
    viseme: Viseme


# Azure viseme IDs name 22 mouth shapes (the same IDs are used for every locale; each
# locale maps its own phonemes onto them). Examples are the en-US IPA phonemes.
AZURE_VISEME_TO_OCULUS: dict[int, Viseme] = {
    0: "sil",
    1: "aa",  # ae, schwa, wedge
    2: "aa",  # open back vowel
    3: "O",
    4: "E",  # open-mid front vowel, near-close back vowel
    5: "RR",  # r-coloured vowel
    6: "I",  # j, i, near-close front vowel
    7: "U",  # w, u
    8: "O",
    9: "aa",  # diphthong "ow"
    10: "O",  # diphthong "oy"
    11: "aa",  # diphthong "ay"
    12: "E",  # h: the mouth follows the next vowel; half-open is the neutral choice
    13: "RR",
    14: "nn",  # l
    15: "SS",  # s, z
    16: "CH",  # sh, ch, j, zh
    17: "TH",  # dh
    18: "FF",  # f, v
    19: "DD",  # d, t, n, th
    20: "kk",  # k, g, ng
    21: "PP",  # p, b, m
}


def map_azure_viseme(viseme_id: int) -> Viseme:
    return AZURE_VISEME_TO_OCULUS.get(viseme_id, "sil")


def from_azure_events(events: Iterable[tuple[int, int]]) -> list[VisemeEvent]:
    """Convert Azure (audio_offset_ms, viseme_id) pairs to Oculus events."""
    return [
        VisemeEvent(max(0, offset), map_azure_viseme(vid)) for offset, vid in events
    ]


def sanitize_visemes(
    events: Iterable[VisemeEvent], duration_ms: int
) -> list[VisemeEvent]:
    """Return a valid timeline for a chunk of `duration_ms` milliseconds of audio.

    Unknown names become "sil", times are clamped to [0, duration_ms] and made strictly
    increasing (when two events share a time the later one wins), repeats of the same
    viseme are merged, the timeline starts at 0 and the mouth closes at the end.
    """
    if duration_ms <= 0:
        return [VisemeEvent(0, "sil")]

    by_time: dict[int, Viseme] = {}
    for event in events:
        name: Viseme = event.viseme if event.viseme in _OCULUS_SET else "sil"
        by_time[min(max(int(event.time_ms), 0), duration_ms)] = name
    items = sorted(by_time.items())
    if not items or items[0][0] > 0:
        items.insert(0, (0, "sil"))

    timeline: list[VisemeEvent] = []
    for time_ms, name in items:
        if timeline and timeline[-1].viseme == name:
            continue
        timeline.append(VisemeEvent(time_ms, name))
    if timeline[-1].viseme != "sil" and timeline[-1].time_ms < duration_ms:
        timeline.append(VisemeEvent(duration_ms, "sil"))
    return timeline


# ---------------------------------------------------------------------------
# Text-based fallback
# ---------------------------------------------------------------------------

_Unit = tuple[Viseme, float]  # (viseme, relative duration)

_VOWEL_WEIGHT = 2.0
_CONSONANT_WEIGHT = 1.0
_COMMA_PAUSE = 2.0
_STOP_PAUSE = 3.0

_VI_DIGRAPHS: dict[str, tuple[Viseme, ...]] = {
    "ngh": ("kk",),
    "ng": ("kk",),
    "nh": ("nn",),
    "ch": ("CH",),
    "tr": ("CH",),
    "th": ("DD",),  # aspirated t in Vietnamese, not the English "th"
    "ph": ("FF",),
    "kh": ("kk",),
    "gh": ("kk",),
    "gi": ("SS",),
    "qu": ("kk", "U"),
}
_VI_SINGLES: dict[str, tuple[Viseme, ...]] = {
    **{c: ("PP",) for c in "bmp"},
    **{c: ("DD",) for c in "tđ"},
    "d": ("SS",),
    "v": ("FF",),
    **{c: ("kk",) for c in "ckqg"},
    **{c: ("SS",) for c in "sx"},
    **{c: ("nn",) for c in "ln"},
    "r": ("RR",),
    "a": ("aa",),
    "e": ("E",),
    **{c: ("I",) for c in "iy"},
    "o": ("O",),
    "u": ("U",),
}

_EN_DIGRAPHS: dict[str, tuple[Viseme, ...]] = {
    "th": ("TH",),
    "sh": ("CH",),
    "ch": ("CH",),
    "ph": ("FF",),
    "ck": ("kk",),
    "ng": ("kk",),
    "wh": ("U",),
    "qu": ("kk", "U"),
}
_EN_SINGLES: dict[str, tuple[Viseme, ...]] = {
    **{c: ("PP",) for c in "bpm"},
    **{c: ("FF",) for c in "fv"},
    **{c: ("DD",) for c in "td"},
    **{c: ("kk",) for c in "kcgq"},
    **{c: ("SS",) for c in "sz"},
    "j": ("CH",),
    "x": ("kk", "SS"),
    **{c: ("nn",) for c in "nl"},
    "r": ("RR",),
    "w": ("U",),
    "a": ("aa",),
    "e": ("E",),
    "i": ("I",),
    "y": ("I",),
    "o": ("O",),
    "u": ("U",),
}
_VOWEL_VISEMES = frozenset({"aa", "E", "I", "O", "U"})


def _base_letters(text: str) -> str:
    """Lower-case and drop accents and tone marks; 'đ' and non-letters are kept."""
    out: list[str] = []
    for char in text.lower():
        if char == "đ":
            out.append(char)
            continue
        decomposed = unicodedata.normalize("NFD", char)
        if decomposed and unicodedata.combining(decomposed[0]) == 0:
            out.append(decomposed[0])
    return "".join(out)


def _units(text: str, language: str) -> list[_Unit]:
    vietnamese = language == "vi"
    digraphs = _VI_DIGRAPHS if vietnamese else _EN_DIGRAPHS
    singles = _VI_SINGLES if vietnamese else _EN_SINGLES
    letters = _base_letters(text)

    def unit(viseme: Viseme) -> _Unit:
        weight = _VOWEL_WEIGHT if viseme in _VOWEL_VISEMES else _CONSONANT_WEIGHT
        return (viseme, weight)

    units: list[_Unit] = []
    i = 0
    while i < len(letters):
        char = letters[i]
        if char in ",;:":
            units.append(("sil", _COMMA_PAUSE))
            i += 1
            continue
        if char in ".!?…\n":
            units.append(("sil", _STOP_PAUSE))
            i += 1
            continue
        matched: tuple[Viseme, ...] | None = None
        length = 1
        for size in (3, 2):
            chunk = letters[i : i + size]
            if len(chunk) == size and chunk in digraphs:
                matched, length = digraphs[chunk], size
                break
        if matched is None:
            matched = singles.get(char)
            if matched is not None and not vietnamese and char == "c":
                if letters[i + 1 : i + 2] in ("e", "i", "y"):
                    matched = ("SS",)
        if matched is not None:
            units.extend(unit(v) for v in matched)
        i += length
    return units


def text_to_visemes(text: str, language: str, duration_ms: int) -> list[VisemeEvent]:
    """Approximate a viseme timeline from spelling, spread over `duration_ms`.

    Used only when the speech provider has no native visemes. The result is always a
    valid timeline (see `sanitize_visemes`).
    """
    if duration_ms <= 0:
        return [VisemeEvent(0, "sil")]
    units = _units(text, language)
    total = sum(weight for _, weight in units)
    if not units or total <= 0:
        return [VisemeEvent(0, "sil")]

    events: list[VisemeEvent] = []
    elapsed = 0.0
    for viseme, weight in units:
        events.append(VisemeEvent(round(duration_ms * elapsed / total), viseme))
        elapsed += weight
    return sanitize_visemes(events, duration_ms)
