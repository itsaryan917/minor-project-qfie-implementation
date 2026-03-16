# Cancer Drug Delivery — Quantum Fuzzy PI Controller

## Overview

This module implements a **Quantum Fuzzy PI Controller** for automatic cancer chemotherapy drug scheduling, inspired by the IFOIMC-based FOPIPOF controller design from:

> **C. Kumar et al.**, *"Indirect fractional-order IMC-based FOPIPOF controller for cancer drug scheduling"*, AEUE (Accepted Paper)

Instead of using a classical fractional-order PI controller, we replace the PI^λ component with a **Quantum Fuzzy Inference Engine (QFIE)** that maps the error and its integral to a drug dosage command.

---

## The Cancer Patient Model

The paper uses a **two-compartment cancer patient model** (Fig. 2 of the paper) with five state variables:

| Variable | Description | Initial Value |
|----------|-------------|---------------|
| **P(t)** | Proliferating cancer cells | 2 × 10¹¹ |
| **Q(t)** | Quiescent cancer cells | 8 × 10¹¹ |
| **D(t)** | Drug concentration (mg/ml) — *controlled variable* | 0 |
| **T(t)** | Toxicity level (mg Day⁻¹/ml) | 0 |
| **Y(t)** | Normal (healthy) cell count affected | 10⁸ |

### Governing Equations

```
dP/dt = (j − k − l)·P + x·Q − r(t)·P          ... (1)  Proliferating cells
dQ/dt = k·P − x·Q                                ... (2)  Quiescent cells
dD/dt = u(t) − θ·D                               ... (4)  Drug concentration
dT/dt = D − β·T                                  ... (6)  Toxicity
dY/dt = σ·Y·(1 − Y/N) − r(t)·Y                  ... (3)  Normal cells
```

Where `r(t) = a · D(t)` is the cell death rate proportional to drug concentration.

### Model Parameters (Table 2)

| Parameter | Description | Value |
|-----------|-------------|-------|
| j | Growth rate of proliferating cells | 0.50 day⁻¹ |
| k | Rate cells change from P to Q | 0.2180 day⁻¹ |
| l | Rate cells stop multiplying | 0.4770 day⁻¹ |
| x | Rate Q cells change into P cells | 0.050 day⁻¹ |
| σ | Speed at which normal cells grow | 0.10 day⁻¹ |
| N | Normal cells' holding capacity | 10⁹ |
| θ | Decay of drugs | 0.270 day⁻¹ |
| a | Cell death rate per unit drug | 8.40 × 10⁻³ day⁻¹ |
| β | Rate of removing toxins | 0.40 |

### Constraints

- Drug concentration hard bound: `0 ≤ D(t) ≤ 50`
- Therapeutic target band: `10 ≤ D(t) ≤ 50`
- Toxicity bound: `T(t) ≤ 100`
- Normal-cell safety threshold: `Y(t) ≥ 10⁶`

---

## Plant Transfer Function

The paper identifies the cancer patient's drug response as a **Fractional First-Order Plus Time Delay (FFOPTD)** model:

```
G(s) = K · e^(−Ls) / (T · s^λ + 1)
```

With identified parameters (Eq. 18):
- **K** = 3.7 (gain)
- **T** = 3.4 (time constant)
- **λ** = 0.989 (fractional order ≈ 1)
- **L** = 0.005 s (time delay — negligible)

The fit achieves **99.26%** accuracy with an error norm of 0.010784.

---

## Controller Design

### Original Paper: IFOIMC Controller

The paper designs an **Indirect Fractional-Order IMC (IFOIMC)** controller:

```
C_IFOIMC(s) = [1/L_1^(−λ)] · [T·e^(−λψ)·s^(λ+1) / (K·e^(Lψ))] · [1 + (1/(T·e^(−λψ))) · s^(−λ)]
               \_________/     \________________________________/    \_____________________________/
                 Filter                    ↕                           PI^λ Controller
```

Optimized parameters: **ψ = −0.0220**, **Kp = 0.9392**, **Ki = 0.2703**

### Our Approach: Quantum Fuzzy PI Controller

We replace the PI^λ component with a **QFIE-based fuzzy PI controller** that:

1. Takes two inputs: **error** (set_point − D) and **integral of error**
2. Uses 25 fuzzy rules (5×5 rule table) to determine a **dose adjustment** Δu
3. Adds feedforward equilibrium dose `u_eq = θ · D_ref` and applies `u = u_eq + Δu`
3. Executes inference on a quantum circuit via the QFIE

This approach offers:
- **Non-linear control** — fuzzy rules can handle the non-linear plant dynamics better
- **Quantum parallelism** — on real quantum hardware, rule evaluation runs in O(R) depth
- **Robustness** — fuzzy logic inherently handles model uncertainty

---

## Fuzzy Configuration

### Input: Error (set_point − D)

| Set | Range | Shape |
|-----|-------|-------|
| NB (Negative Big) | [−15, −15, −8, −3] | Trapezoidal |
| NS (Negative Small) | [−6, −3, 0] | Triangular |
| ZE (Zero) | [−3, 0, 3] | Triangular |
| PS (Positive Small) | [0, 3, 6] | Triangular |
| PB (Positive Big) | [3, 8, 15, 15] | Trapezoidal |

### Input: Integral of Error

| Set | Range | Shape |
|-----|-------|-------|
| NB | [−50, −50, −25, −8] | Trapezoidal |
| NS | [−20, −8, 0] | Triangular |
| ZE | [−8, 0, 8] | Triangular |
| PS | [0, 8, 20] | Triangular |
| PB | [8, 25, 50, 50] | Trapezoidal |

### Output: Dose Adjustment Δu(t)

| Set | Range | Shape |
|-----|-------|-------|
| NB | [−4, −4, −2.4, −1.2] | Trapezoidal |
| NS | [−2, −1, 0] | Triangular |
| ZE | [−0.5, 0, 0.5] | Triangular |
| PS | [0, 1, 2] | Triangular |
| PB | [1.2, 2.4, 4, 4] | Trapezoidal |

### Rule Table (25 Rules)

|  | **integral NB** | **integral NS** | **integral ZE** | **integral PS** | **integral PB** |
|---|---|---|---|---|---|
| **error NB** | VL | VL | VL | LO | LO |
| **error NS** | VL | LO | LO | ME | ME |
| **error ZE** | LO | ME | ME | ME | HI |
| **error PS** | ME | ME | HI | HI | VH |
| **error PB** | HI | HI | VH | VH | VH |

**Convention**: Positive error = drug concentration is below target → need more drug.

---

## Quantum Circuit Structure

For each inference step, the QFIE builds:

```
Qubits:
  error[0..4]     — 5 qubits for error membership (NB, NS, ZE, PS, PB)
  integral[0..4]  — 5 qubits for integral membership
  dosage[0..4]    — 5 qubits for output sets (VL, LO, ME, HI, VH)
  Total: 15 qubits

Encoding:
  Ry(θ) rotations where θ = 2·arcsin(√μ)
  So P(|1⟩) = sin²(θ/2) = μ (membership degree)

Rules:
  25 MCX (multi-controlled-X) gates
  Each maps 2 control qubits → 1 target qubit

Measurement:
  5 output qubits measured → marginal P(|1⟩) gives activation strength
  Centroid defuzzification over dosage universe
```

---

## Scripts

### `cancer_pi_controller_simulation.py`

The main simulation script. Runs the Quantum Fuzzy PI controller on the cancer patient model for three set points:

- **S₁ = 12.08 mg/ml**
- **S₂ = 12.17 mg/ml**
- **S₃ = 11.66 mg/ml**

**What it does:**
1. Initializes the two-compartment cancer model with parameters from Table 2
2. Creates the QFIE-based fuzzy PI controller with 25 rules
3. Runs a closed-loop simulation for 100 days
4. At each step: computes error → QFIE infers dosage → plant model advances
5. Plots drug concentration, dosage, toxicity, error, cancer cells, and normal cells
6. Prints summary metrics (final concentration, IAE, cell counts)

**Run:**
```bash
uv run python cancer_pi_controller_simulation.py
```

### `cancer_pi_controller_comparison.py`

Side-by-side comparison of Classical vs Quantum Fuzzy PI controllers.

**What it compares:**
- Both controllers use **identical** fuzzy rules, membership functions, and plant model
- Classical uses standard Mamdani inference (min-max, centroid)
- Quantum uses QFIE with qubit rotations and MCX gates

**Metrics computed:**
| Metric | Description |
|--------|-------------|
| IAE | Integral of Absolute Error |
| ISE | Integral of Squared Error |
| ITAE | Integral of Time-weighted Absolute Error |
| Settling Time | Time for D(t) to stay within ±2% of set point |
| Overshoot | Maximum overshoot percentage |
| TV | Total Variation of control signal (smoothness, Eq. 35) |
| T_avg | Average toxicity during treatment |
| Per-step time | Inference timing comparison |

**Run:**
```bash
uv run python cancer_pi_controller_comparison.py
```

---

## Connection to the Paper

| Paper Component | Our Implementation |
|----|-----|
| Two-compartment model (Eqs. 1–9) | `CancerPatientModel` class with Euler integration |
| FFOPTD plant G(s) = 3.7e^(−0.005s)/(3.4s^0.989+1) | Simulated directly via differential equations |
| IFOIMC PI^λ controller | Replaced with QFIE-based Fuzzy PI |
| Kp = 0.9392, Ki = 0.2703 | Embedded in fuzzy rule table design |
| Set points S₁, S₂, S₃ | Identical: 12.08, 12.17, 11.66 mg/ml |
| Drug hard bounds [0, 50] and therapeutic band [10, 50] | Hard bounds clipped, therapeutic-band compliance reported |
| Toxicity bound T ≤ 100 | Enforced in model |
| Normal-cell safety threshold Y ≥ 10⁶ | Violation percentage reported |
| Performance metrics (IAE, ISE, TV) | Computed in comparison script |
| Controller comparison (Table 3, Figs. 14–25) | Classical vs Quantum fuzzy comparison |

---

## Key Results (Expected)

1. **Drug concentration** tracks the set point after an initial transient
2. **Proliferating (P) and Quiescent (Q) cells** decrease over the treatment period
3. **Toxicity** remains below the safety threshold (T ≤ 100)
4. **Normal cells (Y)** are preserved during treatment
5. **Classical fuzzy PI** is faster per step (no quantum simulation overhead)
6. **Quantum fuzzy PI** produces similar control quality, demonstrating QFIE feasibility
7. On **real quantum hardware**, the quantum approach would scale better for larger rule bases

---

## Architecture Diagram

```
                     ┌─────────────────────────┐
   Set Point ──(+)──►│  Quantum Fuzzy PI (QFIE) │──── u(t) ────►┌──────────────┐
                │    │  25 rules, 15 qubits     │               │ Cancer Model │
                │    │  error + ∫error → dosage │               │ (P,Q,D,T,Y)  │
                │    └─────────────────────────┘               └──────┬───────┘
                │                                                      │
                └──────────────────── D(t) ◄───────────────────────────┘
                         (feedback)
```
