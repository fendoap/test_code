#!/usr/bin/env python3
"""Auto-generate a pop-style MIDI file.

Creates a pop progression with melody, chords, bass, and drums.
Outputs a standard .mid file without external dependencies.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


TICKS_PER_BEAT = 480
DEFAULT_BPM = 120
DEFAULT_KEY = "C"
DEFAULT_BARS = None
DEFAULT_MINUTES = 3.0
DEFAULT_SEED = 42


NOTE_NAMES = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


@dataclass(frozen=True)
class MidiEvent:
    delta: int
    data: bytes


@dataclass(frozen=True)
class Track:
    name: str
    channel: int
    program: int | None
    events: bytes


def var_len(value: int) -> bytes:
    """Encode value as MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("delta time must be non-negative")
    buffer = value & 0x7F
    value >>= 7
    out = bytearray()
    while value:
        out.insert(0, 0x80 | buffer)
        buffer = value & 0x7F
        value >>= 7
    out.insert(0, buffer)
    return bytes(out)


def note_number(name: str, octave: int) -> int:
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown note name: {name}")
    return 12 * (octave + 1) + NOTE_NAMES[name]


def chord_notes(root: int, quality: str) -> List[int]:
    if quality == "major":
        intervals = [0, 4, 7]
    elif quality == "minor":
        intervals = [0, 3, 7]
    else:
        raise ValueError(f"Unknown chord quality: {quality}")
    return [root + i for i in intervals]


def build_progression(key_root: int) -> List[Tuple[List[int], str]]:
    """Return a richer pop progression with some secondary motion."""
    pattern = [
        (0, "major", "I"),
        (9, "minor", "vi"),
        (5, "major", "IV"),
        (7, "major", "V"),
        (4, "minor", "iii"),
        (9, "minor", "vi"),
        (2, "minor", "ii"),
        (7, "major", "V"),
        (0, "major", "I"),
        (4, "major", "V/vi"),
        (9, "minor", "vi"),
        (5, "major", "IV"),
        (2, "minor", "ii"),
        (7, "major", "V"),
        (0, "major", "I"),
        (0, "major", "I"),
    ]
    progression: List[Tuple[List[int], str]] = []
    for offset, quality, label in pattern:
        progression.append((chord_notes(key_root + offset, quality), label))
    return progression


def add_note_at(
    timeline: List[Tuple[int, int, bytes]],
    start: int,
    note: int,
    velocity: int,
    length: int,
    channel: int,
) -> None:
    status_on = 0x90 | (channel & 0x0F)
    status_off = 0x80 | (channel & 0x0F)
    timeline.append((start, 1, bytes([status_on, note, velocity])))
    timeline.append((start + length, 0, bytes([status_off, note, 0])))


def build_meta_track(bpm: int) -> bytes:
    timeline: List[Tuple[int, int, bytes]] = []
    tempo_us = int(60_000_000 / bpm)
    timeline.append((0, 0, b"\xFF\x51\x03" + tempo_us.to_bytes(3, "big")))
    timeline.append((0, 0, b"\xFF\x58\x04\x04\x02\x18\x08"))
    name = b"Pop Track"
    timeline.append((0, 0, b"\xFF\x03" + bytes([len(name)]) + name))
    timeline.append((0, 2, b"\xFF\x2F\x00"))
    return serialize_timeline(timeline)


def serialize_timeline(timeline: List[Tuple[int, int, bytes]]) -> bytes:
    events: List[MidiEvent] = []
    timeline.sort(key=lambda item: (item[0], item[1]))
    last_time = 0
    for time, _, data in timeline:
        delta = time - last_time
        events.append(MidiEvent(delta, data))
        last_time = time
    track_data = bytearray()
    for event in events:
        track_data.extend(var_len(event.delta))
        track_data.extend(event.data)
    return bytes(track_data)


def build_chord_track(
    key_root: int,
    bars: int,
    channel: int,
    program: int,
) -> Track:
    timeline: List[Tuple[int, int, bytes]] = []
    timeline.append((0, 0, bytes([0xC0 | channel, program])))
    progression = build_progression(key_root)
    bar_ticks = 4 * TICKS_PER_BEAT
    for bar in range(bars):
        bar_start = bar * bar_ticks
        chord_notes_list, _ = progression[bar % len(progression)]
        for note in chord_notes_list:
            add_note_at(timeline, bar_start, note, 70, bar_ticks, channel)
    timeline.append((bars * bar_ticks, 2, b"\xFF\x2F\x00"))
    return Track("Chords", channel, program, serialize_timeline(timeline))


def build_bass_track(
    key_root: int,
    bars: int,
    channel: int,
    program: int,
) -> Track:
    timeline: List[Tuple[int, int, bytes]] = []
    timeline.append((0, 0, bytes([0xC0 | channel, program])))
    progression = build_progression(key_root)
    bar_ticks = 4 * TICKS_PER_BEAT
    half_bar = 2 * TICKS_PER_BEAT
    for bar in range(bars):
        bar_start = bar * bar_ticks
        chord_notes_list, _ = progression[bar % len(progression)]
        root = chord_notes_list[0] - 12
        fifth = root + 7
        add_note_at(timeline, bar_start, root, 80, half_bar, channel)
        add_note_at(timeline, bar_start + half_bar, fifth, 75, half_bar, channel)
    timeline.append((bars * bar_ticks, 2, b"\xFF\x2F\x00"))
    return Track("Bass", channel, program, serialize_timeline(timeline))


def build_melody_track(
    key_root: int,
    bars: int,
    channel: int,
    program: int,
    rng: random.Random,
) -> Track:
    timeline: List[Tuple[int, int, bytes]] = []
    timeline.append((0, 0, bytes([0xC0 | channel, program])))
    bar_ticks = 4 * TICKS_PER_BEAT
    step = TICKS_PER_BEAT // 2
    scale = [0, 2, 4, 5, 7, 9, 11]
    progression = build_progression(key_root)
    for bar in range(bars):
        bar_start = bar * bar_ticks
        chord_notes_list, _ = progression[bar % len(progression)]
        chord_tones = [note % 12 for note in chord_notes_list]
        for step_index in range(8):
            note_start = bar_start + step_index * step
            if step_index in (0, 4):
                degree = rng.choice(chord_tones)
            else:
                degree = (key_root % 12) + rng.choice(scale)
            octave = 5 if step_index % 2 == 0 else 4
            melody_note = degree + 12 * (octave + 1)
            length = step if step_index % 4 != 3 else step * 2
            add_note_at(timeline, note_start, melody_note, 90, length, channel)
    timeline.append((bars * bar_ticks, 2, b"\xFF\x2F\x00"))
    return Track("Melody", channel, program, serialize_timeline(timeline))


def build_drum_track(bars: int) -> Track:
    timeline: List[Tuple[int, int, bytes]] = []
    channel = 9
    bar_ticks = 4 * TICKS_PER_BEAT
    eighth = TICKS_PER_BEAT // 2
    for bar in range(bars):
        bar_start = bar * bar_ticks
        for step in range(8):
            start = bar_start + step * eighth
            add_note_at(timeline, start, 42, 60, eighth, channel)  # Closed hat
        add_note_at(timeline, bar_start, 36, 90, TICKS_PER_BEAT, channel)  # Kick
        add_note_at(timeline, bar_start + 2 * TICKS_PER_BEAT, 36, 85, TICKS_PER_BEAT, channel)
        add_note_at(timeline, bar_start + TICKS_PER_BEAT, 38, 90, TICKS_PER_BEAT, channel)  # Snare
        add_note_at(timeline, bar_start + 3 * TICKS_PER_BEAT, 38, 90, TICKS_PER_BEAT, channel)
    timeline.append((bars * bar_ticks, 2, b"\xFF\x2F\x00"))
    return Track("Drums", channel, None, serialize_timeline(timeline))


def write_midi(path: Path, bpm: int, key: str, bars: int, seed: int) -> None:
    rng = random.Random(seed)
    key_root = note_number(key, 4)
    meta_track = build_meta_track(bpm)
    tracks = [
        meta_track,
        build_chord_track(key_root, bars, channel=0, program=0).events,
        build_bass_track(key_root, bars, channel=1, program=33).events,
        build_melody_track(key_root, bars, channel=2, program=80, rng=rng).events,
        build_drum_track(bars).events,
    ]
    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big")
    header += len(tracks).to_bytes(2, "big") + TICKS_PER_BEAT.to_bytes(2, "big")
    track_chunks = b"".join(b"MTrk" + len(track).to_bytes(4, "big") + track for track in tracks)
    path.write_bytes(header + track_chunks)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a pop-style MIDI file.")
    parser.add_argument("output", type=Path, help="Output .mid path")
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM, help="Tempo in BPM")
    parser.add_argument("--key", type=str, default=DEFAULT_KEY, help="Key (e.g., C, D#, Bb)")
    parser.add_argument(
        "--bars",
        type=int,
        default=DEFAULT_BARS,
        help="Number of bars to generate (overrides --minutes)",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=DEFAULT_MINUTES,
        help="Target length in minutes when --bars is not set",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for melody")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.key not in NOTE_NAMES:
        raise SystemExit(f"Unsupported key: {args.key}")
    if args.bars is not None:
        if args.bars <= 0:
            raise SystemExit("bars must be positive")
        bars = args.bars
    else:
        if args.minutes <= 0:
            raise SystemExit("minutes must be positive")
        seconds_per_bar = (60.0 / args.bpm) * 4
        bars = max(1, round((args.minutes * 60) / seconds_per_bar))
    write_midi(args.output, args.bpm, args.key, bars, args.seed)
    print(f"Wrote MIDI: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
