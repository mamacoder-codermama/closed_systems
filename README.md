# Epistemic Saturation in Closed Systems a simple idea

This project investigates the fundamental epistemological limits of **closed knowledge systems** using symbolic vector models, linear transformations, and semantic equivalence classes. The work formalizes how internal discovery in such systems becomes saturated, and under which conditions new conceptual structures require **external intervention**.

## 🔍 Core Idea: Closed Epistemic Systems

We define a **closed system** as one in which:
- All conceptual knowledge is encoded as symbolic vectors in a finite-dimensional space,
- Chunk our way to see the world, ie, the symbols that compose rules of things and try to apply math on it
    - a vector is a representation of this process of chunking with its components
- Transformations are performed by a fixed linear epistemic matrix \( A \),
- Semantic output is activated through nonlinear functions (e.g., sigmoid),
- Novelty is limited to **combinations** of known concepts.

Without changing the transformation matrix or expanding the vector space, **no genuinely new concept can be discovered**.

## 📐 Linear Tools

- **Concepts** are represented as vectors
- **Knowledge** is represented via a linear transformation \( f(v) = Av \)
- **Activation** is modeled by sigmoid \( \sigma(Av) \) for bounded semantic intensity
- **Discovery** is interpreted as the ability to distinguish new semantic vectors

## 🧠 Semantic Equivalence

We define a semantic equivalence relation:

`v₁ ~ v₂ ⇔ f(v₁) = f(v₂)`

This means the system cannot distinguish `v₁` from `v₂`: they are **epistemically equivalent**.

#### Example:
- `v₃ = [0, 0, 1, 0, ..., 0]`
- `vᴵᴵᴵ = [0.5, 0.5, ..., 0.5]`

If both result in the same semantic activation, then:

`v₃ ~ vᴵᴵᴵ`

## 🧪 Code and Simulation

The repository includes:
- Python simulations of activation, equivalence, and expansion
- Demonstrations of concept collapse
- External interventions (fine-tuning and matrix expansion)
- Visual comparisons of sigmoid-based epistemic outputs

## 📚 Theoretical Context

The approach draws from:
- incompleteness in formal systems
- paradigm shifts and discovery

## 📘 Next Steps

- Extend from linear algebra to **topological epistemics**
- Formalize **emergent concepts** via open sets and limit points

- Feel free to contribute, discuss, or open issues.
- ⚠️ **Status:** work-in-progress – API and maths may change without notice

![status](https://img.shields.io/badge/status-WIP-orange)
![project](https://img.shields.io/badge/type-student_project-lightgrey)

