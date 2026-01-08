#!/usr/bin/env python3
"""Auto-generate a simple pop-style MIDI file.

Creates a catchy 8-bar progression (I-V-vi-IV) with a basic melody
and bass line. Output is a standard .mid file without external deps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


TICKS_PER_BEAT = 480
DEFAULT_BPM = 110
DEFAULT_KEY = "C"


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
    """Return chord tones and label for I-V-vi-IV in given key."""
    progression = [
        (chord_notes(key_root, "major"), "I"),
        (chord_notes(key_root + 7, "major"), "V"),
        (chord_notes(key_root + 9, "minor"), "vi"),
        (chord_notes(key_root + 5, "major"), "IV"),
    ]
    return progression


def add_note_at(timeline: List[Tuple[int, int, bytes]], start: int, note: int, velocity: int, length: int) -> None:
    timeline.append((start, 1, bytes([0x90, note, velocity])))
    timeline.append((start + length, 0, bytes([0x80, note, 0])))


def build_track(bpm: int, key: str) -> bytes:
    events: List[MidiEvent] = []
    timeline: List[Tuple[int, int, bytes]] = []

    tempo_us = int(60_000_000 / bpm)
    timeline.append((0, 0, b"\xFF\x51\x03" + tempo_us.to_bytes(3, "big")))
    timeline.append((0, 0, b"\xFF\x58\x04\x04\x02\x18\x08"))
    timeline.append((0, 0, bytes([0xC0, 0x00])))  # Piano

    key_root = note_number(key, 4)
    progression = build_progression(key_root)

    bar_ticks = 4 * TICKS_PER_BEAT
    chord_tick = bar_ticks

    melody_pattern = [0, 2, 4, 2, 5, 4, 2, 0]
    melody_step = TICKS_PER_BEAT // 2

    for bar in range(8):
        bar_start = bar * bar_ticks
        chord_notes_list, _ = progression[bar % len(progression)]
        root = chord_notes_list[0]
        bass_note = root - 12

        add_note_at(timeline, bar_start, bass_note, 70, chord_tick)

        for step, interval in enumerate(melody_pattern):
            scale_degree = (root + interval) % 12
            melody_note = 60 + scale_degree
            note_start = bar_start + step * melody_step
            add_note_at(timeline, note_start, melody_note, 85, melody_step)

    timeline.append((8 * bar_ticks, 2, b"\xFF\x2F\x00"))

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


def write_midi(path: Path, bpm: int, key: str) -> None:
    track = build_track(bpm, key)
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
    header += (1).to_bytes(2, "big") + TICKS_PER_BEAT.to_bytes(2, "big")
    track_chunk = b"MTrk" + len(track).to_bytes(4, "big") + track
    path.write_bytes(header + track_chunk)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a pop-style MIDI file.")
    parser.add_argument("output", type=Path, help="Output .mid path")
    parser.add_argument("--bpm", type=int, default=DEFAULT_BPM, help="Tempo in BPM")
    parser.add_argument("--key", type=str, default=DEFAULT_KEY, help="Key (e.g., C, D#, Bb)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.key not in NOTE_NAMES:
        raise SystemExit(f"Unsupported key: {args.key}")
    write_midi(args.output, args.bpm, args.key)
    print(f"Wrote MIDI: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
