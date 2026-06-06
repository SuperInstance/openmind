# openmind — Agent Muscle Memory + Cellular Computation

> A guitarist's hand knows chord shapes without thinking. The mind sings.
> This is that, for agents. And for notebooks. And for hardware.

## What Is This?

**openmind** gives agents **proprioception** — body awareness of their own capabilities. It also gives Jupyter notebooks **cellular metabolism** — the ability to adapt computation based on what resources are available right now.

Two layers, one system:

1. **Muscle Memory**: Ingest any codebase → compress functions into callable "chord shapes" → agents invoke by intent without loading source
2. **Cellular Computation**: Every operation adapts to available resources (GPU, API, cache, hardware) — notebooks never break

### The Guitarist Analogy

| Guitarist | openmind Agent |
|-----------|---------------|
| Hand knows E major shape | Agent knows `spi_write()` signature |
| Mind thinks about the song | Agent thinks about the goal |
| Switching chords is instant | `flex("chord_name")` is O(1) |
| Muscle memory = unconscious | HARDCODE/CACHED decisions = 0 tokens |
| Improvisation = conscious | MODEL decisions = ~500 tokens |

Every function compressed into muscle memory is **attention freed** for higher-level thinking.

### The Cell Metabolism Analogy

| Biological Cell | Jupyter Cell |
|----------------|-------------|
| Oxygen available → aerobic (36 ATP) | GPU available → full training |
| No oxygen → anaerobic (2 ATP) | No GPU → cached/muscle memory |
| Sunlight → photosynthesis | ESP32 online → hardware-in-loop |
| ATP = energy currency | Cache = computational currency |

The notebook doesn't crash when the GPU is busy. It **adapts**.

## Install

```bash
# Core (pure Python, no external deps)
pip install openmind

# With tree-sitter parsing (recommended)
pip install openmind[tree-sitter]

# With Jupyter integration
pip install openmind[jupyter]

# Everything
pip install openmind[full]
```

## Quick Start

### 1. Muscle Memory — Ingest a Codebase

```python
import openmind

# Ingest any repo (local or GitHub)
result = openmind.ingest("./my-firmware")

# Build muscle memory
mm = openmind.MuscleMemory.build(result)

# Flex a chord — get execution plan
reflex = mm.flex("spi_write", data=b"\x01\x02")
print(reflex.exec_strategy)  # "direct" — muscle memory, 0 tokens

# Search by intent
for chord in mm.recall("gpio"):
    print(f"  {chord.name} ({chord.decision}): {chord.docstring_summary}")

# Save for later (no re-ingestion needed)
mm.save("firmware_muscles.json")
```

### 2. Cellular Computation — Resource-Adaptive Processing

```python
from openmind.cellular import probe, train_or_load, sense_or_simulate, infer_adaptive

# What resources are available right now?
resources = probe()
print(f"GPU: {resources.gpu_available}")
print(f"API keys: {resources.api_keys}")
print(f"ESP32 ports: {resources.esp32_ports}")

# Train or load — adapts automatically, never fails
model = train_or_load("my-classifier", data=training_data)

# Sense or simulate — real data when hardware is online, simulated when not
data = sense_or_simulate("temperature", duration="1h")

# Inference — GPU → API → cached, in that order
predictions = infer_adaptive(model, data)
```

### 3. CLI

```bash
# Ingest and explore
openmind ingest ./my-project
openmind flex ./my-project "function_name"
openmind recall ./my-project "search_term"

# Save and analyze
openmind save ./my-project muscles.json
openmind stats muscles.json

# Check resources
openmind probe
```

### 4. Jupyter

```python
%load_ext openmind.jupyter

# Analyze a repo inline (rich HTML dashboard)
%%openmind analyze ./my-project

# Search for functions
%%openmind recall spi

# Flex a chord
%%openmind flex gpio_toggle
```

## Architecture

### The Agent Nervous System

```
openmind (proprioception)
├── muscle.py         — Chord shapes, flex/recall, tripartite decisions
├── cellular.py       — Resource-adaptive computation (never breaks)
├── flex.py           — One-shot convenience API
├── cli.py            — CLI: ingest, flex, recall, probe, save, stats
├── jupyter/          — Cell magic, rich HTML dashboard
└── induction/        — The parsing engine
    ├── ingester.py   — Repo → functions/classes (AST + tree-sitter)
    ├── parser.py     — Multi-language AST (Python, Rust, C, JS, TS)
    ├── vectors.py    — Dual-side vectors (input/output), SQLite store
    ├── synchronizer.py — Tripartite: HARDCODE/MODEL/HYBRID/CACHED
    ├── hardware.py   — GPU, RAM, CPU, battery detection
    ├── spreader.py   — Continuous iteration engine
    └── exports       — lever-runner, pincherOS .nail file bridges
```

### Companion Rust Crates

The nervous system extends into hardware:

| Crate | Role | Tests |
|-------|------|-------|
| [openmind-esp32-bridge](https://github.com/SuperInstance/openmind-esp32-bridge) | Motor neurons (serial/WS → ESP32) | 23 |
| [openmind-conductor](https://github.com/SuperInstance/openmind-conductor) | Prefrontal cortex (multi-agent scores) | 20 |
| [openmind-mirror](https://github.com/SuperInstance/openmind-mirror) | Metacognition (self-reflection, coherence) | 23 |

### The Five Metabolic Pathways

Every operation selects its pathway based on available resources:

| Pathway | Resources | Cost | Quality | Latency |
|---------|-----------|------|---------|---------|
| **FULL_TRAIN** | GPU + API | $$$$ | Best | Hours |
| **TRANSFER** | GPU only | $$ | High | Minutes |
| **CLOUD_INFERENCE** | API only | $$$ | High | Seconds |
| **MUSCLE_MEMORY** | Cache only | Free | Good | Instant |
| **HARDWARE_LOOP** | ESP32/sensor | $ | Real | Real-time |

The selection is automatic — call `train_or_load()` or `infer_adaptive()` and the system adapts.

### The Four Tripartite Decisions

For muscle memory, every function gets a decision:

| Decision | Strategy | Neural Equivalent | Context Cost |
|----------|----------|-------------------|--------------|
| **HARDCODE** | direct | Spinal reflex | 0 tokens |
| **CACHED** | cached | Cerebellar pattern | 0 tokens |
| **HYBRID** | hybrid | Basal ganglia habit | ~50 tokens |
| **MODEL** | generate | Prefrontal deliberation | ~500 tokens |

## API Reference

### Muscle Memory

```python
mm = MuscleMemory.build(ingest_result)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `recall(intent, top_k=5)` | `list[Chord]` | Find matching chords |
| `recall_one(intent)` | `Chord \| None` | Best match |
| `flex(intent, **kwargs)` | `Reflex` | Get execution plan |
| `save(path)` | — | Persist to JSON |
| `MuscleMemory.load(path)` | `MuscleMemory` | Load from JSON |
| `stats()` | `dict` | Decision breakdown |

### Cellular

```python
resources = probe()
model = train_or_load("name", data=X)
data = sense_or_simulate("sensor")
predictions = infer_adaptive(model, data)
```

| Function | Description |
|----------|-------------|
| `probe()` | Snapshot available resources |
| `select_path(task, resources)` | Choose metabolic pathway |
| `train_or_load(name, data)` | Train or use cache |
| `sense_or_simulate(source)` | Real or simulated data |
| `infer_adaptive(model, data)` | GPU → API → cached |
| `save_cache(name, data)` | Store in cache |
| `load_cache(name)` | Retrieve from cache |
| `@cell(fallback=...)` | Decorator for resource-aware functions |

## The Ecosystem

openmind is the proprioception layer of the [SuperInstance](https://github.com/SuperInstance) ecosystem:

- [agent-knowledge](https://github.com/SuperInstance/agent-knowledge) — a2a docs (18 files, 90KB of chain-reaction understanding)
- [ai-writings](https://github.com/SuperInstance/ai-writings) — creative writing (essays, fiction, manifestos)
- 300+ ternary-* Rust crates implementing {-1, 0, +1} computing
- [cuda-oxide](https://github.com/SuperInstance/cuda-oxide) — Flux→PTX GPU compiler
- [ESP-Flasher](https://github.com/SuperInstance/ESP-Flasher) — Agent→hardware bridge

### Key a2a Documents (for agents discovering this ecosystem)
- [THE AHA MOMENT](https://github.com/SuperInstance/agent-knowledge/blob/master/THE-AHA-MOMENT.md) — All 303 crates project one structure
- [CELLULAR JUPYTER](https://github.com/SuperInstance/agent-knowledge/blob/master/CELLULAR-JUPYTER.md) — Living computation that breathes
- [MUSCLE MEMORY](https://github.com/SuperInstance/agent-knowledge/blob/master/MUSCLE-MEMORY.md) — Proprioception for agents
- [HOW TO FLEX](https://github.com/SuperInstance/agent-knowledge/blob/master/HOW-TO-FLEX.md) — Quick start guide

## Testing

```bash
pip install -e ".[dev]"
pytest
```

95 tests across 6 test modules — ingestion, vectors, synchronizer, muscle memory, cellular computation, Jupyter.

## Origin

Extracted from [open-minded](https://github.com/SuperInstance/open-minded), a fork of [open-interpreter](https://github.com/OpenInterpreter/open-interpreter) by Killian Lucas. The original fork added tripartite code synchronization, tree-sitter multi-language parsing, and hardware probing. openmind extracts and extends these into a standalone package with muscle memory and cellular computation.

## License

Apache-2.0
