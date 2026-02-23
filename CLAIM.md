# NeuroGuard: Priority Claim

> **A Geometric SVM-Based Early Warning Framework for Catastrophic Forgetting in Continual Learning**

---

## Author

- **GitHub:** [Aurov84](https://github.com/Aurov84)
- **Date of Claim:** February 22, 2026

---

## Abstract

We propose **NeuroGuard**, a novel framework that provides **proactive early warning** for catastrophic forgetting in continual learning. Unlike existing approaches (e.g., EWC, PackNet, Experience Replay) that mitigate forgetting *during or after* training, NeuroGuard detects potential knowledge conflicts **before** new training begins, by leveraging the geometric properties of individual neurons and Support Vector Machine (SVM) theory.

The core idea is to model each neuron's learned knowledge as a **geometric solution space** defined by a hyperplane orientation and a margin region. When new data arrives, NeuroGuard evaluates whether the new knowledge is geometrically compatible with existing knowledge — enabling researchers and practitioners to anticipate and prevent catastrophic forgetting rather than react to it.

---

## 1. Theoretical Foundation

### 1.1 Neuron as a Geometric Classifier

A single neuron with *n* inputs performs a linear classification in *n*-dimensional space. Its *n* weights define the **normal vector** of an (*n*−1)-dimensional hyperplane, and the bias determines the **offset** of this hyperplane from the origin.

During training, the optimization process adjusts:

- The **normal vector** (weight vector) → controls the *orientation* of the decision boundary
- The **bias** → controls the *position* of the decision boundary

### 1.2 Margin as Knowledge Representation

Drawing from SVM theory, when a neuron successfully learns a linearly separable task, there exists a **margin** — the region between the closest data points (support vectors) on either side of the hyperplane. The training process centers the hyperplane to maximize this margin.

We propose that this margin constitutes a **geometric representation of the neuron's learned knowledge**:

- The **normal vector direction** encodes *what* the neuron has learned (the classification rule)
- The **margin width** encodes *how robustly* the neuron has learned it (the solution space tolerance)

Together, they define a **safe region** — the set of all hyperplane configurations that preserve the neuron's learned knowledge.

---

## 2. Proposed Method

### 2.1 Knowledge State Capture

After training converges (i.e., loss reaches a sufficiently low value), NeuroGuard records two quantities per neuron:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Hyperplane normal vector | **w** | Orientation of the learned decision boundary |
| Half-margin width | ½ *M* | Radius of the geometric safe region |

### 2.2 Conflict Detection (Early Warning)

When new training data is introduced in a continual learning scenario, it induces a *candidate* hyperplane with its own normal vector **w'** and margin *M'*. NeuroGuard performs two tests:

**Test 1 — Orientation Compatibility:**
Compute the cosine similarity between the existing normal vector **w** and the candidate normal vector **w'**. A low similarity indicates that the new task requires a fundamentally different decision boundary orientation.

**Test 2 — Margin Intersection:**
Determine whether the margin regions *M* and *M'* have a non-empty intersection. If the margins do not overlap, no single hyperplane configuration can satisfy both the old and new knowledge simultaneously.

### 2.3 Warning Criteria

| Condition | Interpretation | Risk Level |
|-----------|---------------|------------|
| High similarity + margin overlap | Compatible knowledge | ✅ Safe |
| High similarity + no overlap | Positional conflict | ⚠️ Warning |
| Low similarity + margin overlap | Partial conflict | ⚠️ Warning |
| Low similarity + no overlap | Fundamental conflict | 🚨 Critical |

If a **critical conflict** is detected, forcing the neuron to learn the new knowledge will push the hyperplane outside the old knowledge's solution space, **destroying previously learned representations**.

---

## 3. Key Contributions

1. **Pre-training early warning:** To our knowledge, this is the first method that detects catastrophic forgetting risk *before* new training begins, rather than mitigating it during or after training.

2. **Geometric interpretability:** The framework provides a clear, mathematically grounded explanation of *why* forgetting occurs at the individual neuron level — the new knowledge's solution space is geometrically incompatible with the old knowledge's solution space.

3. **Neuron-level granularity:** By operating at the level of individual neurons, the method can pinpoint *which specific neurons* are at risk, enabling targeted intervention strategies.

4. **Theoretical hypothesis:** We hypothesize that the geometric destruction of a neuron's solution space by incompatible new knowledge is a fundamental mechanism underlying catastrophic forgetting.

---

## 4. Relationship to Existing Work

| Method | When it acts | Mechanism |
|--------|-------------|-----------|
| EWC (Elastic Weight Consolidation) | During training | Penalizes changes to important weights |
| PackNet | After training | Prunes and freezes learned weights |
| Experience Replay | During training | Replays old data to retain knowledge |
| Progressive Neural Networks | Before training | Adds new capacity, freezes old |
| **NeuroGuard (this work)** | **Before training** | **Detects geometric conflicts proactively** |

---

## 5. Priority Statement

This document is committed to a public GitHub repository to establish a verifiable timestamp for the priority of this discovery. The Git commit hash and associated timestamp serve as cryptographic evidence of the date of this claim.

**Commit history and timestamps are publicly auditable at:**
`https://github.com/Aurov84/NeuroGuard`

---

## License

This work is released under the [MIT License](./LICENSE).
