"""
Comparison: Classical vs Quantum Fuzzy Drone Altitude Stabilization
====================================================================
Runs both controllers on the same scenario, measures:
  - Control accuracy  (final error, settling time, overshoot)
  - Per-step timing   (wall-clock time per inference call)
  - Total simulation time
  - Output agreement  (MAE between the two controllers)

Produces a side-by-side plot and prints a summary table.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from src.QFIE.FuzzyEngines import QuantumFuzzyEngine, trimf, trapmf
from src.QFIE.ClassicalFuzzyEngine import ClassicalFuzzyEngine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared configuration (identical for both engines)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALTITUDE_ERROR_UNIVERSE = np.linspace(-50, 50, 200)
ERROR_RATE_UNIVERSE     = np.linspace(-20, 20, 200)
MOTOR_THRUST_UNIVERSE   = np.linspace(-100, 100, 200)

ALT_SETS = [
    trapmf(ALTITUDE_ERROR_UNIVERSE, [-50, -50, -10, 0]),
    trimf(ALTITUDE_ERROR_UNIVERSE,  [-10,   0,  10]),
    trapmf(ALTITUDE_ERROR_UNIVERSE, [  0,  10,  50, 50]),
]
RATE_SETS = [
    trapmf(ERROR_RATE_UNIVERSE, [-20, -20, -5, 0]),
    trimf(ERROR_RATE_UNIVERSE,  [ -5,   0,  5]),
    trapmf(ERROR_RATE_UNIVERSE, [  0,   5, 20, 20]),
]
THRUST_SETS = [
    trapmf(MOTOR_THRUST_UNIVERSE, [-100, -100, -50, -10]),
    trimf(MOTOR_THRUST_UNIVERSE,  [ -20,    0,  20]),
    trapmf(MOTOR_THRUST_UNIVERSE, [  10,   50, 100, 100]),
]

RULES = [
    'if altitude_error is zero and error_rate is zero then motor_thrust is zero',
    'if altitude_error is pos and error_rate is zero then motor_thrust is pos_big',
    'if altitude_error is neg and error_rate is zero then motor_thrust is neg_big',
    'if altitude_error is zero and error_rate is pos then motor_thrust is pos_big',
    'if altitude_error is zero and error_rate is neg then motor_thrust is neg_big',
    'if altitude_error is pos and error_rate is pos then motor_thrust is pos_big',
    'if altitude_error is neg and error_rate is neg then motor_thrust is neg_big',
    'if altitude_error is pos and error_rate is neg then motor_thrust is zero',
    'if altitude_error is neg and error_rate is pos then motor_thrust is zero',
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build both engines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_quantum_engine():
    engine = QuantumFuzzyEngine(verbose=False, encoding='linear')
    engine.input_variable("altitude_error", ALTITUDE_ERROR_UNIVERSE)
    engine.input_variable("error_rate", ERROR_RATE_UNIVERSE)
    engine.output_variable("motor_thrust", MOTOR_THRUST_UNIVERSE)
    engine.add_input_fuzzysets("altitude_error", ["neg", "zero", "pos"], ALT_SETS)
    engine.add_input_fuzzysets("error_rate", ["neg", "zero", "pos"], RATE_SETS)
    engine.add_output_fuzzysets("motor_thrust", ["neg_big", "zero", "pos_big"], THRUST_SETS)
    engine.set_rules(RULES)
    return engine


def build_classical_engine():
    engine = ClassicalFuzzyEngine(verbose=False)
    engine.input_variable("altitude_error", ALTITUDE_ERROR_UNIVERSE)
    engine.input_variable("error_rate", ERROR_RATE_UNIVERSE)
    engine.output_variable("motor_thrust", MOTOR_THRUST_UNIVERSE)
    engine.add_input_fuzzysets("altitude_error", ["neg", "zero", "pos"], ALT_SETS)
    engine.add_input_fuzzysets("error_rate", ["neg", "zero", "pos"], RATE_SETS)
    engine.add_output_fuzzysets("motor_thrust", ["neg_big", "zero", "pos_big"], THRUST_SETS)
    engine.set_rules(RULES)
    return engine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulation runner (generic — takes any engine)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_simulation(engine, engine_type, error0, velocity0, steps, dt=0.05, n_shots=1024):
    """Run the drone physics loop using the given engine.

    Returns dict with errors, velocities, thrusts, step_times, total_time.
    """
    error = error0
    velocity = velocity0

    errors = []
    velocities = []
    thrusts = []
    step_times = []

    total_start = time.perf_counter()

    for t in range(steps):
        safe_error = float(np.clip(error, -50, 50))
        safe_rate  = float(np.clip(velocity, -20, 20))
        crisp_inputs = {'altitude_error': safe_error, 'error_rate': safe_rate}

        t0 = time.perf_counter()

        if engine_type == 'quantum':
            engine.build_inference_qc(crisp_inputs, draw_qc=False, distributed=False)
            thrust, _ = engine.execute(n_shots=n_shots)
        else:
            thrust, _ = engine.infer(crisp_inputs)

        t1 = time.perf_counter()

        thrust = float(thrust)
        step_times.append(t1 - t0)

        errors.append(error)
        velocities.append(velocity)
        thrusts.append(thrust)

        acceleration = error - thrust - 0.5 * velocity
        velocity += acceleration * dt
        error    += velocity * dt

    total_time = time.perf_counter() - total_start

    return {
        'errors': np.array(errors),
        'velocities': np.array(velocities),
        'thrusts': np.array(thrusts),
        'step_times': np.array(step_times),
        'total_time': total_time,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def settling_time(errors, threshold=1.0):
    """Index of the first step after which |error| stays below threshold."""
    for i in range(len(errors) - 1, -1, -1):
        if abs(errors[i]) > threshold:
            return i + 1
    return 0


def max_overshoot(errors, initial_sign):
    """Max magnitude of error on the opposite side of the initial sign."""
    if initial_sign < 0:
        overshoot = np.max(errors)   # crossed to positive
    else:
        overshoot = -np.min(errors)  # crossed to negative
    return max(0.0, overshoot)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    STEPS = 100
    ERROR0 = -15.0
    VEL0 = 0.0
    N_SHOTS = 1024

    print("=" * 70)
    print("  COMPARISON: Classical vs Quantum Fuzzy Drone Altitude Controller")
    print("=" * 70)
    print(f"  Initial error  : {ERROR0:+.1f} cm")
    print(f"  Initial rate   : {VEL0:+.1f} cm/s")
    print(f"  Steps          : {STEPS}")
    print(f"  Quantum shots  : {N_SHOTS}")
    print("=" * 70)

    # ── Build engines ────────────────────────────────────────────────
    print("\nBuilding classical engine...")
    c_engine = build_classical_engine()

    print("Building quantum engine...")
    q_engine = build_quantum_engine()

    # ── Run simulations ──────────────────────────────────────────────
    print("\nRunning CLASSICAL simulation...")
    c_res = run_simulation(c_engine, 'classical', ERROR0, VEL0, STEPS, n_shots=N_SHOTS)

    print("Running QUANTUM simulation...")
    q_res = run_simulation(q_engine, 'quantum', ERROR0, VEL0, STEPS, n_shots=N_SHOTS)

    # ── Metrics ──────────────────────────────────────────────────────
    initial_sign = np.sign(ERROR0)

    c_settle = settling_time(c_res['errors'])
    q_settle = settling_time(q_res['errors'])

    c_overshoot = max_overshoot(c_res['errors'], initial_sign)
    q_overshoot = max_overshoot(q_res['errors'], initial_sign)

    c_final = abs(c_res['errors'][-1])
    q_final = abs(q_res['errors'][-1])

    c_avg_step = np.mean(c_res['step_times']) * 1000     # ms
    q_avg_step = np.mean(q_res['step_times']) * 1000     # ms

    c_med_step = np.median(c_res['step_times']) * 1000
    q_med_step = np.median(q_res['step_times']) * 1000

    mae = np.mean(np.abs(c_res['thrusts'] - q_res['thrusts']))

    speedup = q_avg_step / c_avg_step if c_avg_step > 0 else float('inf')

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"{'METRIC':<35} {'CLASSICAL':>15} {'QUANTUM':>15}")
    print("─" * 70)
    print(f"{'Final |error| (cm)':<35} {c_final:>15.4f} {q_final:>15.4f}")
    print(f"{'Settling time (step, ±1 cm)':<35} {c_settle:>15d} {q_settle:>15d}")
    print(f"{'Max overshoot (cm)':<35} {c_overshoot:>15.4f} {q_overshoot:>15.4f}")
    print(f"{'Avg step time (ms)':<35} {c_avg_step:>15.4f} {q_avg_step:>15.4f}")
    print(f"{'Median step time (ms)':<35} {c_med_step:>15.4f} {q_med_step:>15.4f}")
    print(f"{'Total simulation time (s)':<35} {c_res['total_time']:>15.4f} {q_res['total_time']:>15.4f}")
    print(f"{'Thrust MAE (C vs Q)':<35} {mae:>15.4f} {'—':>15}")
    print(f"{'Classical speedup factor':<35} {'1.00x':>15} {f'{speedup:.1f}x slower':>15}")
    print("─" * 70)

    # ── Time-complexity analysis ─────────────────────────────────────
    n_input_vars = 2
    n_sets_per_var = 3
    n_rules = len(RULES)
    n_universe = len(MOTOR_THRUST_UNIVERSE)

    print("\n" + "=" * 70)
    print("  TIME-COMPLEXITY ANALYSIS")
    print("=" * 70)
    print(f"""
  Let:
    V = number of input variables     = {n_input_vars}
    S = fuzzy sets per variable        = {n_sets_per_var}
    R = number of rules                = {n_rules}
    N = universe discretization points = {n_universe}
    K = quantum shots (samples)        = {N_SHOTS}

  ┌─────────────────┬──────────────────────────────────┬─────────────────────────────────┐
  │ Stage           │ CLASSICAL                        │ QUANTUM                         │
  ├─────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Fuzzification   │ O(V × S)                         │ O(V × S)  [same]                │
  │ Rule evaluation │ O(R × V)                         │ O(R)  gate placement            │
  │                 │ (R rules, each checks V inputs)  │ MCX gates; depth = R            │
  │ Circuit build   │ —                                │ O(V×S + R)  qubits + gates      │
  │ Execution       │ —                                │ O(K × 2^Q)  simulation*         │
  │ Defuzzification │ O(S_out × N)                     │ O(S_out × N)  [same]            │
  ├─────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ TOTAL per step  │ O(R×V + S_out×N)                 │ O(K × 2^Q + S_out×N)           │
  │                 │ ≈ O({n_rules}×{n_input_vars} + {n_sets_per_var}×{n_universe})              │ ≈ O({N_SHOTS} × 2^9 + {n_sets_per_var}×{n_universe})          │
  │                 │ ≈ O({n_rules * n_input_vars + n_sets_per_var * n_universe})                          │ ≈ O({N_SHOTS * (2**9) + n_sets_per_var * n_universe})                     │
  └─────────────────┴──────────────────────────────────┴─────────────────────────────────┘

  * Q = total qubits = V×S + S_out = {n_input_vars}×{n_sets_per_var} + {n_sets_per_var} = {n_input_vars * n_sets_per_var + n_sets_per_var} qubits.
    On a classical SIMULATOR, the cost is O(K × 2^Q) per shot because
    it must track 2^Q amplitudes (statevector). This is why quantum
    simulation is slower on classical hardware.

  KEY INSIGHT:
  ────────────
  On REAL quantum hardware, rule evaluation is O(R) in circuit DEPTH
  (not 2^Q), because all qubits operate in parallel. As the number of
  rules and input variables scale up (e.g. 100+ rules, 10+ inputs),
  classical grows as O(R × V) while quantum stays O(R) in depth.

  The quantum ADVANTAGE emerges when:
    • R × V  >>  circuit depth × gate time
    • i.e., many variables and many rules

  For our small system ({n_rules} rules, {n_input_vars} inputs), classical
  is faster because the simulation overhead O(2^Q) dominates.
""")

    # ── Plotting ─────────────────────────────────────────────────────
    time_axis = np.arange(STEPS)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- (0,0) Altitude error ---
    ax = axes[0, 0]
    ax.plot(time_axis, c_res['errors'], linewidth=2, label='Classical', color='royalblue')
    ax.plot(time_axis, q_res['errors'], linewidth=2, label='Quantum', color='crimson', linestyle='--')
    ax.axhline(0, color='black', linestyle=':', alpha=0.4)
    ax.set_title("Altitude Error (goal → 0)")
    ax.set_ylabel("Error (cm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (0,1) Motor thrust ---
    ax = axes[0, 1]
    ax.plot(time_axis, c_res['thrusts'], linewidth=2, label='Classical', color='royalblue')
    ax.plot(time_axis, q_res['thrusts'], linewidth=2, label='Quantum', color='crimson', linestyle='--')
    ax.axhline(0, color='black', linestyle=':', alpha=0.4)
    ax.set_title("Motor Thrust Adjustment")
    ax.set_ylabel("Thrust (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (1,0) Per-step timing ---
    ax = axes[1, 0]
    ax.plot(time_axis, c_res['step_times'] * 1000, linewidth=1.5, label='Classical', color='royalblue')
    ax.plot(time_axis, q_res['step_times'] * 1000, linewidth=1.5, label='Quantum', color='crimson')
    ax.set_title("Per-Step Inference Time")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Time Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # --- (1,1) Thrust difference ---
    ax = axes[1, 1]
    diff = q_res['thrusts'] - c_res['thrusts']
    ax.bar(time_axis, diff, color='darkorange', alpha=0.7, width=1.0)
    ax.axhline(0, color='black', linestyle=':', alpha=0.4)
    ax.set_title(f"Thrust Difference (Quantum − Classical)  MAE={mae:.2f}%")
    ax.set_ylabel("Δ Thrust (%)")
    ax.set_xlabel("Time Step")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Classical vs Quantum Fuzzy — Drone Altitude Stabilization",
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150)
    print(f"\nPlot saved to comparison_results.png")
    plt.show()


if __name__ == "__main__":
    main()
