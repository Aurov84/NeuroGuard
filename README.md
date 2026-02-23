# NeuroGuard

**A Geometric SVM-Based Early Warning System for Catastrophic Forgetting in Continual Learning**

---

## What is NeuroGuard?

NeuroGuard is a novel method that **predicts catastrophic forgetting before it happens**.

Existing approaches (EWC, PackNet, Experience Replay) try to fix forgetting *during or after* training. NeuroGuard takes a fundamentally different approach: it analyzes the geometric properties of individual neurons to detect knowledge conflicts **before** new training begins.

### Core Insight

Every trained neuron defines a hyperplane in weight space. That hyperplane has two measurable properties:

- **Normal vector** (the weight vector) — encodes *what* the neuron learned
- **Half-margin** (distance to nearest support point) — encodes *how much room* the neuron has before its knowledge breaks

When new knowledge arrives, NeuroGuard compares these two properties between the old and new states. If the margin regions don't overlap, the new knowledge **will** destroy the old — and NeuroGuard warns you before any damage occurs.

---

## How It Works

```
1. Train neuron on Task A → record normal vector + margin
2. Clone neuron (mirror) → train mirror on Task B
3. Compare:
   - Normal vector cosine similarity → direction compatible?
   - Margin interval intersection   → solution space compatible?
4. Result:
   - Overlap exists    → safe to learn
   - No overlap        → WARNING: catastrophic forgetting will occur
```

### Conflict Detection Table

| Condition | Risk Level |
|-----------|-----------|
| High similarity + margin overlap | ✅ Safe |
| High similarity + no overlap | ⚠️ Warning — positional conflict |
| Low similarity + margin overlap | ⚠️ Warning — partial conflict |
| Low similarity + no overlap | 🚨 Critical — fundamental conflict |

---

## Comparison with Existing Methods

| Method | When It Acts | Approach | Extra Cost |
|--------|-------------|----------|------------|
| EWC | During training | Penalizes weight changes | O(n²) Fisher matrix |
| PackNet | After training | Prunes and freezes weights | Pruning passes |
| Experience Replay | During training | Replays old data | Memory for old data |
| Progressive Nets | Before training | Adds new capacity | Network grows |
| **NeuroGuard** | **Before training** | **Geometric conflict detection** | **O(n) per neuron** |

---

## Quick Start

### Requirements

- Python 3.x
- matplotlib (usually pre-installed with Python)
- tkinter (built into Python)

### Run

```bash
python neuroguard_viz.py
```

### Experiment Workflow

1. **Clone** — create a mirror from the real neuron (preserves optimal initialization)
2. **Select Task** — choose AND or OR logic gate
3. **Train** — the mirror learns the new task
4. **Compute Memory** — calculate hyperplane slope and half-margin
5. **Conflict Detection** — compare old vs. new knowledge geometrically
6. **Write Back or Discard** — accept or reject the new knowledge

---

## Why Clone Instead of Random Init?

The real neuron holds the current optimal state. Cloning it as initialization ensures:

- The hyperplane direction stays **consistent** with prior knowledge
- Avoids the case where random initialization produces a **reversed** normal vector (cosine similarity = −1), which classifies correctly but is geometrically incompatible

---

## Engineering Advantages

NeuroGuard is designed as an **external monitoring system** — zero modification to the model itself:

- **Zero invasion** — no changes to model architecture or training code
- **Zero extra parameters** — the normal vector *is* the weight vector (already stored); only 1 float per neuron for margin
- **Lightweight** — conflict detection is O(n) per neuron, takes microseconds even at n=4000
- **Pluggable** — can be attached or removed at any time without affecting the model

Storage can be as simple as an external JSON file mapping each neuron's margin:

```json
{
  "layer_0": {
    "neuron_0": { "half_margin": 0.342, "timestamp": "2026-02-22T20:00:00Z" },
    "neuron_1": { "half_margin": 0.518, "timestamp": "2026-02-22T20:00:00Z" }
  }
}
```

The normal vector doesn't need separate storage — it's the weight vector itself.

---

## Repository Structure

```
NeuroGuard/
├── README.md            ← You are here
├── CLAIM.md             ← Priority claim with theoretical foundation
├── neuroguard_viz.py    ← Interactive visualization platform
└── LICENSE              ← MIT License
```

---

## Priority Claim

This method was first documented on **February 22, 2026**.

See [CLAIM.md](./CLAIM.md) for the full theoretical foundation and priority statement.

- **Commit hash:** 9450bce
- **Release:** [v0.1.0-claim](https://github.com/Aurov84/NeuroGuard/releases/tag/v0.1.0-claim)

---

## Status

This is an early-stage research project. Current status:

- [x] Core theory formulated
- [x] Priority claim established
- [x] 2D proof-of-concept with AND/OR logic gates
- [x] Interactive visualization platform
- [ ] Validation on standard benchmarks (MNIST, CIFAR)
- [ ] Multi-layer network experiments
- [ ] Comparison experiments with EWC, PackNet
- [ ] Paper submission

---

## License

[MIT License](./LICENSE)

---

## Author

**Zhiqiang Chen**
GitHub: [@Aurov84](https://github.com/Aurov84)
