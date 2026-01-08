# MIDI Generator (Rich Pop / Neo-Soul)

This project generates a rich, harmonically complex pop/neo-soul style MIDI file.
The output is a SMF Type 1 `.mid` file at **120 BPM**, **4/4**, and **90 bars** (about 3 minutes).

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python midi_gen.py --out output.mid --seed 42 --key C --mode major
```

Outputs:

- `output.mid`
- `output.progression.json` (chord progression and tags for verification)

## Colab

Use `colab_midi_generator.ipynb`, then run the cells in order to:

1. Install dependencies
2. Generate `output.mid`
3. Download the file
