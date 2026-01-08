#!/usr/bin/env python3
"""Generate a rich pop/jazz-inspired MIDI arrangement."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import mido


TICKS_PER_BEAT = 480
BPM = 120
BARS = 90
BEATS_PER_BAR = 4
MIN_NOTE_TICKS = TICKS_PER_BEAT // 4
DEFAULT_KEY = "C"
DEFAULT_MODE = "major"

SECTION_PLAN = [
    ("Intro", 8),
    ("A", 16),
    ("B", 16),
    ("Chorus", 16),
    ("Interlude", 8),
    ("A'", 8),
    ("Chorus'", 16),
    ("Outro", 2),
]

REQUIRED_TAGS = {
    "SEC_DOM": 8,
    "BORROW": 6,
    "TRITONE": 3,
    "DIM": 4,
    "ALT": 6,
    "TONICIZE": 2,
}

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NOTE_TO_PC = {
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

QUALITY_INTERVALS = {
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "sus4": [0, 5, 7, 10],
    "6/9": [0, 4, 7, 9, 14],
    "add9": [0, 4, 7, 14],
}

TENSION_INTERVALS = {
    "9": 14,
    "b9": 13,
    "#9": 15,
    "11": 17,
    "#11": 18,
    "13": 21,
    "b13": 20,
}


@dataclass(frozen=True)
class Chord:
    roman: str
    symbol: str
    root_pc: int
    quality: str
    tensions: Sequence[str]
    bass_pc: int | None
    tags: Sequence[str]
    duration_beats: int


@dataclass
class ProgressionEvent:
    section: str
    chord: Chord


def pc_to_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def midi_note(pc: int, octave: int) -> int:
    return 12 * (octave + 1) + (pc % 12)


def chord_pitch_classes(chord: Chord) -> List[int]:
    intervals = QUALITY_INTERVALS[chord.quality][:]
    for tension in chord.tensions:
        intervals.append(TENSION_INTERVALS[tension])
    pcs = sorted({(chord.root_pc + interval) % 12 for interval in intervals})
    return pcs


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def ensure_min_duration(ticks: int) -> int:
    return max(ticks, MIN_NOTE_TICKS)


def choose_voicing(chord: Chord, prev: Sequence[int] | None) -> List[int]:
    pcs = chord_pitch_classes(chord)
    if len(pcs) < 4:
        pcs.append((chord.root_pc + 7) % 12)
    if len(pcs) > 6:
        pcs = pcs[:6]
    target_notes: List[int] = []
    base = 60
    for idx, pc in enumerate(pcs):
        target = base + idx * 3
        if prev and idx < len(prev):
            target = prev[idx]
        candidates = [midi_note(pc, octave) for octave in range(2, 6)]
        note = min(candidates, key=lambda n: abs(n - target))
        note = clamp(note, midi_note(pc, 2), midi_note(pc, 5))
        target_notes.append(note)
    target_notes.sort()
    return target_notes


def choose_pad_notes(chord: Chord) -> List[int]:
    pcs = chord_pitch_classes(chord)
    chosen = []
    for pc in pcs:
        note = midi_note(pc, 4)
        if 48 <= note <= 84:
            chosen.append(note)
    if not chosen:
        chosen.append(midi_note(chord.root_pc, 4))
    if len(chosen) < 3:
        chosen.extend(chosen[:3])
    return sorted(set(chosen))[:4]


def build_chord(roman: str, root_pc: int, quality: str, tensions: Sequence[str], tags: Sequence[str], duration: int, bass_pc: int | None = None) -> Chord:
    tension_str = "".join(tensions)
    symbol = f"{pc_to_name(root_pc)}{quality}{tension_str}"
    return Chord(
        roman=roman,
        symbol=symbol,
        root_pc=root_pc,
        quality=quality,
        tensions=list(tensions),
        bass_pc=bass_pc,
        tags=list(tags),
        duration_beats=duration,
    )


def diatonic_degree(key_root: int, degree: int, mode: str) -> int:
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    minor_scale = [0, 2, 3, 5, 7, 8, 10]
    scale = major_scale if mode == "major" else minor_scale
    return (key_root + scale[degree - 1]) % 12


def secondary_dominant(key_root: int, target_degree: int, mode: str, altered: bool, duration: int) -> Chord:
    target_pc = diatonic_degree(key_root, target_degree, mode)
    root_pc = (target_pc + 7) % 12
    tensions = ["b9", "#9"] if altered else ["9"]
    tags = ["D", "SEC_DOM"] + (["ALT"] if altered else [])
    return build_chord(f"V/{target_degree}", root_pc, "7", tensions, tags, duration)


def tritone_sub(key_root: int, target_degree: int, mode: str, duration: int) -> Chord:
    target_pc = diatonic_degree(key_root, target_degree, mode)
    root_pc = (target_pc + 7 + 6) % 12
    return build_chord(f"SubV/{target_degree}", root_pc, "7", ["b9", "#11"], ["D", "TRITONE", "ALT"], duration)


def borrowed_chord(key_root: int, roman: str, root_pc: int, quality: str, duration: int) -> Chord:
    return build_chord(roman, root_pc, quality, ["9"], ["CT", "BORROW"], duration)


def dim_passing(key_root: int, roman: str, root_pc: int, duration: int) -> Chord:
    return build_chord(roman, root_pc, "dim7", [], ["CT", "DIM"], duration)


def tonicize_block(key_root: int, target_degree: int, mode: str) -> List[Chord]:
    target_pc = diatonic_degree(key_root, target_degree, mode)
    ii_pc = (target_pc + 2) % 12
    v_pc = (target_pc + 7) % 12
    vi_pc = (target_pc + 9) % 12
    chords = [
        build_chord(f"ii/{target_degree}", ii_pc, "m7", ["9"], ["PD", "TONICIZE"], 2),
        build_chord(f"V/{target_degree}", v_pc, "7", ["b9", "13"], ["D", "TONICIZE", "ALT"], 2),
        build_chord(f"I/{target_degree}", target_pc, "maj7", ["9", "13"], ["T", "TONICIZE"], 4),
        build_chord(f"vi/{target_degree}", vi_pc, "m7", ["9"], ["T", "TONICIZE"], 4),
        build_chord(f"ii/{target_degree}", ii_pc, "m7", ["11"], ["PD", "TONICIZE"], 2),
        build_chord(f"V/{target_degree}", v_pc, "7", ["#9", "b13"], ["D", "TONICIZE", "ALT"], 2),
    ]
    return chords


FUNCTION_TRANSITIONS = {
    "T": (["T", "PD", "CT"], [0.35, 0.5, 0.15]),
    "PD": (["D", "CT", "PD"], [0.55, 0.25, 0.2]),
    "D": (["T", "CT"], [0.8, 0.2]),
    "CT": (["PD", "D", "T"], [0.4, 0.4, 0.2]),
}


def choose_next_function(current: str, rng: random.Random) -> str:
    options, weights = FUNCTION_TRANSITIONS[current]
    return rng.choices(options, weights=weights, k=1)[0]


def pick_tonic_chord(key_root: int, mode: str, rng: random.Random, duration: int) -> Chord:
    options = [
        build_chord("I", diatonic_degree(key_root, 1, mode), "maj7", ["9", "13"], ["T"], duration),
        build_chord("I", diatonic_degree(key_root, 1, mode), "6/9", [], ["T"], duration),
        build_chord("I", diatonic_degree(key_root, 1, mode), "add9", [], ["T"], duration),
        build_chord("vi", diatonic_degree(key_root, 6, mode), "m7", ["9"], ["T"], duration),
        build_chord("iii", diatonic_degree(key_root, 3, mode), "m7", ["9"], ["T"], duration),
    ]
    return rng.choice(options)


def pick_predominant_chord(key_root: int, mode: str, rng: random.Random, duration: int) -> Chord:
    options = [
        build_chord("ii", diatonic_degree(key_root, 2, mode), "m7", ["11"], ["PD"], duration),
        build_chord("IV", diatonic_degree(key_root, 4, mode), "maj7", ["9"], ["PD"], duration),
        build_chord("ii", diatonic_degree(key_root, 2, mode), "m7b5", ["11"], ["PD"], duration),
        borrowed_chord(key_root, "iv", (key_root + 5) % 12, "m7", duration),
        borrowed_chord(key_root, "bVI", (key_root + 8) % 12, "maj7", duration),
    ]
    return rng.choice(options)


def pick_dominant_chord(key_root: int, mode: str, rng: random.Random, duration: int) -> Chord:
    target_degree = rng.choice([2, 5, 6])
    options = [
        build_chord("V", diatonic_degree(key_root, 5, mode), "7", ["9"], ["D"], duration),
        build_chord("V", diatonic_degree(key_root, 5, mode), "7", ["b9", "#9"], ["D", "ALT"], duration),
        secondary_dominant(key_root, target_degree, mode, altered=rng.random() < 0.7, duration=duration),
        tritone_sub(key_root, target_degree, mode, duration=duration),
    ]
    return rng.choice(options)


def pick_chromatic_chord(key_root: int, mode: str, rng: random.Random, duration: int) -> Chord:
    options = [
        borrowed_chord(key_root, "bVII", (key_root + 10) % 12, "7", duration),
        borrowed_chord(key_root, "bIII", (key_root + 3) % 12, "maj7", duration),
        borrowed_chord(key_root, "bVI", (key_root + 8) % 12, "maj7", duration),
        dim_passing(key_root, "#ivo7", (key_root + 6) % 12, duration),
    ]
    return rng.choice(options)


def generate_section(
    section: str,
    length: int,
    key_root: int,
    mode: str,
    rng: random.Random,
    include_tonicize: bool,
) -> List[Chord]:
    chords: List[Chord] = []
    bar = 0
    function_state = "T"
    while bar < length:
        use_two_beat = section in {"B", "Chorus", "Chorus'"} and rng.random() < 0.6
        if include_tonicize and bar == 4:
            chords.extend(tonicize_block(key_root, 5, mode))
            bar += 4
            function_state = "T"
            continue
        if use_two_beat and bar + 1 <= length:
            chords.append(pick_predominant_chord(key_root, mode, rng, duration=2))
            chords.append(pick_dominant_chord(key_root, mode, rng, duration=2))
            function_state = "T"
        else:
            if function_state == "T":
                chord = pick_tonic_chord(key_root, mode, rng, duration=4)
            elif function_state == "PD":
                chord = pick_predominant_chord(key_root, mode, rng, duration=4)
            elif function_state == "D":
                chord = pick_dominant_chord(key_root, mode, rng, duration=4)
            else:
                chord = pick_chromatic_chord(key_root, mode, rng, duration=4)
            chords.append(chord)
            function_state = choose_next_function(function_state, rng)
        bar += 1
    return chords


def generate_progression(key_root: int, mode: str, seed: int) -> List[ProgressionEvent]:
    rng = random.Random(seed)
    attempts = 0
    while True:
        attempts += 1
        events: List[ProgressionEvent] = []
        counts = {tag: 0 for tag in REQUIRED_TAGS}
        tonicize_used = 0
        for section, length in SECTION_PLAN:
            include_tonicize = tonicize_used < 2 and section in {"B", "Chorus"}
            chords = generate_section(section, length, key_root, mode, rng, include_tonicize)
            if include_tonicize:
                tonicize_used += 1
            for chord in chords:
                for tag in chord.tags:
                    if tag in counts:
                        counts[tag] += 1
                events.append(ProgressionEvent(section=section, chord=chord))
        if all(counts[tag] >= minimum for tag, minimum in REQUIRED_TAGS.items()):
            return events
        rng = random.Random(seed + attempts)


def render_log(events: List[ProgressionEvent]) -> None:
    current_section = None
    buffer: List[str] = []
    for event in events:
        if event.section != current_section:
            if buffer:
                print("  " + " | ".join(buffer))
                buffer = []
            current_section = event.section
            print(f"[{current_section}]")
        chord = event.chord
        tags = ",".join(chord.tags)
        buffer.append(f"{chord.roman}({chord.symbol})[{tags}]")
    if buffer:
        print("  " + " | ".join(buffer))


def add_notes_timeline(
    timeline: List[tuple[int, int, mido.Message]],
    start: int,
    duration: int,
    notes: Sequence[int],
    velocity: int,
    channel: int,
) -> None:
    duration = ensure_min_duration(duration)
    for note in notes:
        timeline.append((start, 0, mido.Message("note_on", note=note, velocity=velocity, time=0, channel=channel)))
    for note in notes:
        timeline.append((start + duration, 1, mido.Message("note_off", note=note, velocity=0, time=0, channel=channel)))


def timeline_to_track(timeline: List[tuple[int, int, mido.Message]]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    timeline.sort(key=lambda item: (item[0], item[1]))
    last_time = 0
    for tick, _, message in timeline:
        delta = tick - last_time
        track.append(message.copy(time=delta))
        last_time = tick
    return track


def build_tracks(events: List[ProgressionEvent], seed: int) -> mido.MidiFile:
    rng = random.Random(seed)
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("track_name", name="Arrangement", time=0))
    midi.tracks.append(meta)

    piano_timeline: List[tuple[int, int, mido.Message]] = [
        (0, 0, mido.Message("program_change", program=0, time=0, channel=0))
    ]
    bass_timeline: List[tuple[int, int, mido.Message]] = [
        (0, 0, mido.Message("program_change", program=33, time=0, channel=1))
    ]
    pad_timeline: List[tuple[int, int, mido.Message]] = [
        (0, 0, mido.Message("program_change", program=89, time=0, channel=2))
    ]
    drum_timeline: List[tuple[int, int, mido.Message]] = []

    midi.tracks.extend([mido.MidiTrack(), mido.MidiTrack(), mido.MidiTrack(), mido.MidiTrack()])
    prev_voicing: List[int] | None = None
    current_tick = 0

    for event in events:
        chord = event.chord
        duration_ticks = chord.duration_beats * TICKS_PER_BEAT
        voicing = choose_voicing(chord, prev_voicing)
        prev_voicing = voicing
        velocity = rng.randint(70, 100)
        pattern_roll = rng.random()
        if pattern_roll < 0.4:
            add_notes_timeline(piano_timeline, current_tick, duration_ticks, voicing, velocity, channel=0)
        elif pattern_roll < 0.75:
            half = ensure_min_duration(duration_ticks // 2)
            add_notes_timeline(piano_timeline, current_tick, half, voicing, velocity, channel=0)
            add_notes_timeline(piano_timeline, current_tick + half, half, voicing, velocity - 10, channel=0)
        else:
            step = ensure_min_duration(duration_ticks // 4)
            for step_index in range(4):
                note = [rng.choice(voicing)]
                add_notes_timeline(piano_timeline, current_tick + step_index * step, step, note, velocity - 15, channel=0)

        bass_pc = chord.bass_pc if chord.bass_pc is not None else chord.root_pc
        bass_note = midi_note(bass_pc, 2)
        bass_note = clamp(bass_note, 28, 48)
        add_notes_timeline(bass_timeline, current_tick, duration_ticks, [bass_note], rng.randint(60, 95), channel=1)

        pad_notes = choose_pad_notes(chord)
        add_notes_timeline(pad_timeline, current_tick, duration_ticks, pad_notes, rng.randint(50, 85), channel=2)

        drum_step = ensure_min_duration(duration_ticks // 2)
        add_notes_timeline(drum_timeline, current_tick, drum_step, [42], 60, channel=9)
        add_notes_timeline(drum_timeline, current_tick + drum_step, drum_step, [42], 55, channel=9)
        add_notes_timeline(drum_timeline, current_tick, duration_ticks, [36], 80, channel=9)
        add_notes_timeline(
            drum_timeline, current_tick + duration_ticks // 2, duration_ticks // 2, [38], 85, channel=9
        )

        current_tick += duration_ticks

    midi.tracks[1] = timeline_to_track(piano_timeline)
    midi.tracks[2] = timeline_to_track(bass_timeline)
    midi.tracks[3] = timeline_to_track(pad_timeline)
    midi.tracks[4] = timeline_to_track(drum_timeline)

    return midi


def progression_to_json(events: List[ProgressionEvent], seed: int, key: str, mode: str) -> dict:
    return {
        "seed": seed,
        "key": key,
        "mode": mode,
        "bpm": BPM,
        "bars": BARS,
        "sections": [
            {
                "section": event.section,
                "roman": event.chord.roman,
                "symbol": event.chord.symbol,
                "duration_beats": event.chord.duration_beats,
                "tags": event.chord.tags,
            }
            for event in events
        ],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a rich pop/jazz MIDI file.")
    parser.add_argument("--out", type=Path, default=Path("output.mid"), help="Output .mid path")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (omit for new random output each run)",
    )
    parser.add_argument("--key", type=str, default=DEFAULT_KEY, help="Key (e.g., C, Db, F#)")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["major", "minor"], help="Mode")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.key not in NOTE_TO_PC:
        raise SystemExit(f"Unsupported key: {args.key}")
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    key_root = NOTE_TO_PC[args.key]
    events = generate_progression(key_root, args.mode, seed)
    total_beats = sum(event.chord.duration_beats for event in events)
    expected_beats = BARS * BEATS_PER_BAR
    if total_beats != expected_beats:
        raise SystemExit(f"Progression length mismatch: {total_beats} beats (expected {expected_beats})")
    render_log(events)
    midi = build_tracks(events, seed)
    midi.save(args.out)
    progression_path = args.out.with_suffix(".progression.json")
    progression_path.write_text(json.dumps(progression_to_json(events, seed, args.key, args.mode), indent=2))
    print(f"Seed: {seed}")
    print(f"Wrote MIDI: {args.out}")
    print(f"Saved progression: {progression_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
