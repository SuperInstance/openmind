# openmind — Agent Muscle Memory

> A guitarist's hand knows chord shapes without thinking. The mind sings.
> This is that, for agents.

## What Is This?

**openmind** gives an AI agent **proprioception** — a body's awareness of its own capabilities. When an agent controls ESP32 devices, compiles GPU kernels, or orchestrates distributed systems, it shouldn't burn its limited context window on *how* to toggle GPIO pin 15. It should just **flex** the muscle.

The induction engine ingests a codebase and compresses every function into a **"chord shape"** — a callable unit the agent invokes by intent, not by source. The agent's conscious attention (context window) stays free for what matters: the melody, not the fingering.

### The Guitarist Analogy

| Guitarist | Agent |
|-----------|-------|
| Hand knows E major shape | Agent knows `spi_write()` signature |
| Mind thinks about the song | Agent thinks about the goal |
| Switching chords is instant | Invoking a chord is O(1) |
| Learning a new chord takes attention | Generating novel code burns context |
| Muscle memory = unconscious | HARDCODE/CACHED decisions |
| Improvisation = conscious | MODEL decisions |

Every function compressed into muscle memory is **attention freed**. A 300-crate ecosystem with 6,000+ functions becomes 6,000 chords the agent doesn't have to think about.

## Install

```bash
# Core (pure Python, no tree-sitter)
pip install openmind

# With multi-language parsing (recommended)
pip install openmind[tree-sitter]

# With Jupyter integration
pip install openmind[jupyter]

# Everything
pip install openmind[full]
```

## Quick Start

### CLI

```bash
# Ingest a repository
openmind ingest ./my-esp32-firmware

# Flex a chord — get the execution plan
openmind flex ./firmware "spi_write"

# Search for matching chords
openmind recall ./firmware "gpio"

# Save muscle memory for later (no re-ingestion)
openmind save ./firmware firmware_memory.json

# Check statistics
openmind stats firmware_memory.json

# Hardware probe
openmind probe
```

### Python API

```python
import openmind

# Ingest a codebase
result = openmind.ingest("./my-esp32-firmware")

# Build muscle memory
mm = openmind.MuscleMemory.build(result)

# Recall a chord by intent
chord = mm.recall_one("spi_write")
print(f"{chord.name}: {chord.decision}")  # "hardcode" — muscle memory

# Flex — get the full execution plan
reflex = mm.flex("spi_write", data=b"\x01\x02")
print(reflex.exec_strategy)  # "direct" — call it directly, no thinking needed

# Search by fuzzy intent
for chord in mm.recall("gpio"):
    print(f"  {chord.name} ({chord.decision}): {chord.docstring_summary}")

# Save for later
mm.save("firmware_muscles.json")
```

### Jupyter

```python
%load_ext openmind.jupyter

# Analyze a repo inline (renders rich HTML dashboard)
%%openmind analyze ./firmware

# Search for functions
%%openmind recall spi

# Flex a chord
%%openmind flex gpio_toggle
```

## Architecture

### The Four Strategies (Tripartite Synchronizer)

The synchronizer decides **how much thinking** each function needs:

| Decision | Strategy | Analogy | When |
|----------|----------|---------|------|
| **HARDCODE** | `direct` | Muscle memory | Hot path, deterministic, many callers |
| **CACHED** | `cached` | Replay | Pre-computed, read-only, edge device |
| **HYBRID** | `hybrid` | Chord + solo | Mostly cached, model fallback |
| **MODEL** | `generate` | Improvisation | Novel, creative, untested |

### Decision Factors

Three inputs drive the decision (the "tripartite"):

1. **Hardware** — GPU available? RAM? Battery? Edge device?
2. **Application** — Latency required? Safety critical? Scale?
3. **User** — Manual control? Creativity? Consistency preference?

A function with 5+ callers, tests, and a safety-critical context → HARDCODE (muscle memory).
An untested function called once → MODEL (the agent thinks about it).

### The Pipeline

```
Codebase → Ingest → Parse (AST/tree-sitter) → Functions/Classes
                                              ↓
                              MuscleMemory.build()
                                              ↓
                    Tripartite Synchronizer decides per-function
                                              ↓
                              Chord shapes compressed + indexed
                                              ↓
                              Agent calls flex("intent", args)
                                              ↓
                              Reflex returned (execution plan)
```

## API Reference

### `MuscleMemory`

```python
mm = MuscleMemory.build(ingest_result)
```

| Method | Description |
|--------|-------------|
| `recall(intent, top_k=5)` | Find chords matching an intent |
| `recall_one(intent)` | Best match or None |
| `flex(intent, **kwargs)` | Get execution plan (Reflex) |
| `save(path)` | Persist to JSON |
| `MuscleMemory.load(path)` | Load from JSON |
| `stats()` | Decision breakdown, test coverage |

### `Chord`

A compressed function shape — the agent's muscle memory unit.

| Field | Description |
|-------|-------------|
| `name` | Function name (the "chord name") |
| `module` | Module path |
| `signature` | Type signature |
| `decision` | HARDCODE/MODEL/HYBRID/CACHED |
| `intent_keywords` | Words that trigger this chord |
| `docstring_summary` | First line of docstring |
| `has_tests` | Whether it's tested |
| `call_count` | How many functions it calls |
| `called_by` | Functions that call it |
| `matches(query)` | Relevance score 0.0–1.0 |

### `Reflex`

The execution plan returned by `flex()`.

| Field | Description |
|-------|-------------|
| `chord` | The matched Chord |
| `exec_strategy` | "direct" / "cached" / "generate" / "hybrid" |
| `confidence` | 0.0–1.0 (higher if tested) |
| `cached_result` | Pre-computed result (CACHED) |
| `generator_hint` | Prompt for LLM (MODEL) |

## Use Cases

### 1. ESP32 Agent with Body Memory

```python
# One-time setup: ingest the ESP32 firmware
result = openmind.ingest("./esp32-firmware")
mm = openmind.MuscleMemory.build(result)
mm.save("esp32_muscles.json")

# At runtime: the agent has "body awareness"
mm = openmind.MuscleMemory.load("esp32_muscles.json")

# Agent wants to blink an LED — muscle memory handles it
reflex = mm.flex("gpio_toggle", pin=2)
if reflex.exec_strategy == "direct":
    execute_directly(reflex.chord, pin=2)  # No LLM needed
else:
    code = generate_with_llm(reflex.generator_hint)  # Improvise
```

### 2. Multi-Repo Code Search

```python
# Build muscle memory for multiple repos
repos = ["./firmware", "./drivers", "./protocols"]
for repo in repos:
    result = openmind.ingest(repo)
    mm = openmind.MuscleMemory.build(result)
    mm.save(f"muscles/{os.path.basename(repo)}.json")

# Agent searches across all repos for "i2c"
for path in glob("muscles/*.json"):
    mm = openmind.MuscleMemory.load(path)
    for chord in mm.recall("i2c"):
        print(f"{path}: {chord.name} — {chord.docstring_summary}")
```

### 3. Jupyter Notebook Analysis

```python
%load_ext openmind.jupyter

%%openmind analyze ./ternary-core
# Shows rich HTML dashboard with decision breakdown

%%openmind recall vector
# Shows matching functions with signatures

%%openmind flex tdot
# Shows execution strategy: HYBRID, 50% confidence
```

## Relation to the Ternary Ecosystem

openmind was born from analyzing the [SuperInstance ternary fleet](https://github.com/SuperInstance) — 300+ Rust crates implementing {-1, 0, +1} computing. The induction engine discovered a **spectral isomorphism** (>0.97 cosine similarity) proving all repos project the same mathematical structure.

In the oxide stack architecture:
- **open-parallel** (async runtime) → agent's nervous system
- **pincher** (intent compiler) → agent's language center
- **flux-core** (bytecode VM) → agent's interpreter
- **cuda-oxide** (PTX backend) → agent's GPU motor cortex
- **cudaclaw** (GPU execution) → agent's hands
- **openmind** (induction engine) → agent's **proprioception** — knowing where its hands are without looking

## Origin

Extracted from [open-minded](https://github.com/SuperInstance/open-minded), a fork of [open-interpreter](https://github.com/OpenInterpreter/open-interpreter) by Killian Lucas. The original fork added a tripartite code synchronizer, tree-sitter multi-language parser, and hardware probe — but these innovations were trapped inside a broken LLM chat chain. openmind **rips out the good parts** and makes them standalone.

## Development

```bash
git clone https://github.com/SuperInstance/open-mind-standalone
cd open-mind-standalone
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0
