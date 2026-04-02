"""
Comparison: Classical vs Quantum Fuzzy PI — Cancer Drug Delivery
=================================================================
Runs both the Classical Fuzzy PI and Quantum Fuzzy PI controllers
on the same cancer patient model, measuring:

  - Drug concentration tracking (IAE, ISE, ITAE)
  - Settling time and overshoot
  - Toxicity levels
  - Cancer cell reduction (P, Q cells)
  - Normal cell preservation (Y cells)
  - Per-step inference timing
  - Total variation of control signal (smoothness)

Produces side-by-side comparison plots and a summary table.
"""

import os
import time

import numpy as np
# Use a non-GUI backend by default to avoid Tkinter shutdown errors.
import matplotlib


def _env_flag(name, default="0"):
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return int(default)


SHOW_PLOTS = _env_flag("CANCER_SHOW_PLOTS", "0")
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from src.QFIE.FuzzyEngines import QuantumFuzzyEngine, trimf, trapmf
from src.QFIE.ClassicalFuzzyEngine import ClassicalFuzzyEngine
from cancer_pi_controller_simulation import (
    CancerPatientModel, PARAMS, SET_POINTS, D_MIN, D_MAX, T_MAX,
    THERAPEUTIC_D_MIN, Y_MIN_SAFE, DOSAGE_MAX, SIM_DAYS, DT,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared fuzzy configuration (identical for both engines)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERROR_UNIVERSE    = np.linspace(-15, 15, 200)
INTEGRAL_UNIVERSE = np.linspace(-50, 50, 200)
DOSAGE_UNIVERSE   = np.linspace(-4, 4, 200)

ERROR_SETS = [
    trapmf(ERROR_UNIVERSE, [-15, -15, -8, -3]),    # NB
    trimf(ERROR_UNIVERSE,  [-6, -3, 0]),            # NS
    trimf(ERROR_UNIVERSE,  [-3, 0, 3]),             # ZE
    trimf(ERROR_UNIVERSE,  [0, 3, 6]),              # PS
    trapmf(ERROR_UNIVERSE, [3, 8, 15, 15]),         # PB
]

INTEGRAL_SETS = [
    trapmf(INTEGRAL_UNIVERSE, [-50, -50, -25, -8]),  # NB
    trimf(INTEGRAL_UNIVERSE,  [-20, -8, 0]),          # NS
    trimf(INTEGRAL_UNIVERSE,  [-8, 0, 8]),            # ZE
    trimf(INTEGRAL_UNIVERSE,  [0, 8, 20]),            # PS
    trapmf(INTEGRAL_UNIVERSE, [8, 25, 50, 50]),       # PB
]

DOSAGE_SETS = [
    trapmf(DOSAGE_UNIVERSE, [-4, -4, -2.4, -1.2]),
    trimf(DOSAGE_UNIVERSE,  [-2.0, -1.0, 0.0]),
    trimf(DOSAGE_UNIVERSE,  [-0.5, 0.0, 0.5]),
    trimf(DOSAGE_UNIVERSE,  [0.0, 1.0, 2.0]),
    trapmf(DOSAGE_UNIVERSE, [1.2, 2.4, 4, 4]),
]

ERROR_NAMES    = ["NB", "NS", "ZE", "PS", "PB"]
INTEGRAL_NAMES = ["NB", "NS", "ZE", "PS", "PB"]
DOSAGE_NAMES   = ["NB", "NS", "ZE", "PS", "PB"]

RULES = [
    # error NB
    'if error is NB and integral is NB then dosage is NB',
    'if error is NB and integral is NS then dosage is NB',
    'if error is NB and integral is ZE then dosage is NB',
    'if error is NB and integral is PS then dosage is NS',
    'if error is NB and integral is PB then dosage is ZE',
    # error NS
    'if error is NS and integral is NB then dosage is NB',
    'if error is NS and integral is NS then dosage is NS',
    'if error is NS and integral is ZE then dosage is NS',
    'if error is NS and integral is PS then dosage is ZE',
    'if error is NS and integral is PB then dosage is PS',
    # error ZE
    'if error is ZE and integral is NB then dosage is NB',
    'if error is ZE and integral is NS then dosage is NS',
    'if error is ZE and integral is ZE then dosage is ZE',
    'if error is ZE and integral is PS then dosage is PS',
    'if error is ZE and integral is PB then dosage is PB',
    # error PS
    'if error is PS and integral is NB then dosage is NS',
    'if error is PS and integral is NS then dosage is ZE',
    'if error is PS and integral is ZE then dosage is PS',
    'if error is PS and integral is PS then dosage is PS',
    'if error is PS and integral is PB then dosage is PB',
    # error PB
    'if error is PB and integral is NB then dosage is ZE',
    'if error is PB and integral is NS then dosage is PS',
    'if error is PB and integral is ZE then dosage is PB',
    'if error is PB and integral is PS then dosage is PB',
    'if error is PB and integral is PB then dosage is PB',
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build engines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_quantum_engine():
    engine = QuantumFuzzyEngine(verbose=False, encoding='linear')
    engine.input_variable("error", ERROR_UNIVERSE)
    engine.input_variable("integral", INTEGRAL_UNIVERSE)
    engine.output_variable("dosage", DOSAGE_UNIVERSE)
    engine.add_input_fuzzysets("error", ERROR_NAMES, ERROR_SETS)
    engine.add_input_fuzzysets("integral", INTEGRAL_NAMES, INTEGRAL_SETS)
    engine.add_output_fuzzysets("dosage", DOSAGE_NAMES, DOSAGE_SETS)
    engine.set_rules(RULES)
    return engine


def build_classical_engine():
    engine = ClassicalFuzzyEngine(verbose=False)
    engine.input_variable("error", ERROR_UNIVERSE)
    engine.input_variable("integral", INTEGRAL_UNIVERSE)
    engine.output_variable("dosage", DOSAGE_UNIVERSE)
    engine.add_input_fuzzysets("error", ERROR_NAMES, ERROR_SETS)
    engine.add_input_fuzzysets("integral", INTEGRAL_NAMES, INTEGRAL_SETS)
    engine.add_output_fuzzysets("dosage", DOSAGE_NAMES, DOSAGE_SETS)
    engine.set_rules(RULES)
    return engine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulation runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_simulation(engine, engine_type, set_point, days=SIM_DAYS, dt=DT, n_shots=1024):
    """Run the cancer model closed-loop with the given fuzzy engine.

    Returns dict with all recorded signals and timing information.
    """
    steps = int(days / dt)
    model = CancerPatientModel()
    integral_error = 0.0

    time_arr = np.zeros(steps)
    drug_conc = np.zeros(steps)
    drug_dose_arr = np.zeros(steps)
    toxicity = np.zeros(steps)
    p_cells = np.zeros(steps)
    q_cells = np.zeros(steps)
    y_cells = np.zeros(steps)
    errors = np.zeros(steps)
    step_times = np.zeros(steps)

    total_start = time.perf_counter()

    for t in range(steps):
        time_arr[t] = t * dt
        error = set_point - model.D
        errors[t] = error

        # Accumulate integral
        integral_error += error * dt
        integral_error = float(np.clip(integral_error, -50, 50))
        safe_error = float(np.clip(error, -15, 15))
        safe_integral = integral_error
        crisp_inputs = {'error': safe_error, 'integral': safe_integral}

        t0 = time.perf_counter()

        if engine_type == 'quantum':
            engine.build_inference_qc(crisp_inputs, draw_qc=False, distributed=False)
            delta_dose, _ = engine.execute(n_shots=n_shots)
        else:
            delta_dose, _ = engine.infer(crisp_inputs)

        t1 = time.perf_counter()
        step_times[t] = t1 - t0

        dosage = PARAMS['theta'] * set_point + float(delta_dose)
        dosage = float(np.clip(dosage, 0, DOSAGE_MAX))

        # Advance plant model
        model.step(dosage, dt)

        drug_conc[t] = model.D
        drug_dose_arr[t] = dosage
        toxicity[t] = model.T
        p_cells[t] = model.P
        q_cells[t] = model.Q
        y_cells[t] = model.Y

    total_time = time.perf_counter() - total_start

    return {
        'time': time_arr, 'drug_conc': drug_conc, 'drug_dose': drug_dose_arr,
        'toxicity': toxicity, 'p_cells': p_cells, 'q_cells': q_cells,
        'y_cells': y_cells, 'errors': errors, 'step_times': step_times,
        'total_time': total_time,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(results, set_point, dt):
    """Compute performance metrics from simulation results."""
    errors = results['errors']
    drug_conc = results['drug_conc']
    drug_dose = results['drug_dose']
    toxicity = results['toxicity']

    iae = np.sum(np.abs(errors)) * dt
    ise = np.sum(errors ** 2) * dt
    itae = np.sum(np.abs(errors) * results['time']) * dt

    # Settling time: first time drug_conc stays within ±2% of set_point
    tol = 0.02 * set_point
    settled = np.abs(drug_conc - set_point) < tol
    settle_idx = len(errors)
    for i in range(len(settled) - 1, -1, -1):
        if not settled[i]:
            settle_idx = i + 1
            break
    settling_time = settle_idx * dt if settle_idx < len(errors) else float('inf')

    # Overshoot
    overshoot = max(0, np.max(drug_conc) - set_point)
    overshoot_pct = (overshoot / set_point) * 100 if set_point > 0 else 0

    # Total variation of control signal (Eq. 35 in paper)
    tv = np.sum(np.abs(np.diff(drug_dose)))

    # Average and final toxicity
    avg_toxicity = np.mean(toxicity)
    final_toxicity = toxicity[-1]
    therapeutic_band_pct = 100 * np.mean((drug_conc >= THERAPEUTIC_D_MIN) & (drug_conc <= D_MAX))
    toxicity_violation_pct = 100 * np.mean(toxicity > T_MAX)
    y_safety_violation_pct = 100 * np.mean(results['y_cells'] < Y_MIN_SAFE)

    return {
        'IAE': iae, 'ISE': ise, 'ITAE': itae,
        'settling_time': settling_time,
        'overshoot': overshoot, 'overshoot_pct': overshoot_pct,
        'TV': tv,
        'avg_toxicity': avg_toxicity, 'final_toxicity': final_toxicity,
        'therapeutic_band_pct': therapeutic_band_pct,
        'toxicity_violation_pct': toxicity_violation_pct,
        'y_safety_violation_pct': y_safety_violation_pct,
        'final_P': results['p_cells'][-1],
        'final_Q': results['q_cells'][-1],
        'final_Y': results['y_cells'][-1],
        'final_D': drug_conc[-1],
        'avg_step_ms': np.mean(results['step_times']) * 1000,
        'median_step_ms': np.median(results['step_times']) * 1000,
        'total_time': results['total_time'],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    N_SHOTS = _env_int("CANCER_QFIE_SHOTS", 128)
    SP_NAME = 'S1'              # Primary comparison set point
    SP = SET_POINTS[SP_NAME]

    print("=" * 72)
    print("  COMPARISON: Classical vs Quantum Fuzzy PI — Cancer Drug Delivery")
    print("=" * 72)
    print(f"  Set point ({SP_NAME})  : {SP:.2f} mg/ml")
    print(f"  Duration         : {SIM_DAYS} days")
    print(f"  Time step        : {DT} days")
    print(f"  Quantum shots    : {N_SHOTS}")
    print("=" * 72)

    # ── Build engines ────────────────────────────────────────────────
    print("\nBuilding classical fuzzy PI engine...")
    c_engine = build_classical_engine()

    print("Building quantum fuzzy PI engine...")
    q_engine = build_quantum_engine()

    # ── Run simulations ──────────────────────────────────────────────
    print("\nRunning CLASSICAL fuzzy PI simulation...")
    c_res = run_simulation(c_engine, 'classical', SP, n_shots=N_SHOTS)

    print("Running QUANTUM fuzzy PI simulation...")
    q_res = run_simulation(q_engine, 'quantum', SP, n_shots=N_SHOTS)

    # ── Compute metrics ──────────────────────────────────────────────
    c_met = compute_metrics(c_res, SP, DT)
    q_met = compute_metrics(q_res, SP, DT)

    # ── Print summary table ──────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"{'METRIC':<40} {'CLASSICAL':>15} {'QUANTUM':>15}")
    print("─" * 72)
    print(f"{'Final Drug Conc (mg/ml)':<40} {c_met['final_D']:>15.4f} {q_met['final_D']:>15.4f}")
    print(f"{'IAE':<40} {c_met['IAE']:>15.4f} {q_met['IAE']:>15.4f}")
    print(f"{'ISE':<40} {c_met['ISE']:>15.4f} {q_met['ISE']:>15.4f}")
    print(f"{'ITAE':<40} {c_met['ITAE']:>15.4f} {q_met['ITAE']:>15.4f}")
    print(f"{'Settling Time (days, ±2%)':<40} {c_met['settling_time']:>15.2f} {q_met['settling_time']:>15.2f}")
    print(f"{'Overshoot (%)':<40} {c_met['overshoot_pct']:>15.4f} {q_met['overshoot_pct']:>15.4f}")
    print(f"{'Control Signal TV':<40} {c_met['TV']:>15.4f} {q_met['TV']:>15.4f}")
    print(f"{'Therapeutic Band Time (%)':<40} {c_met['therapeutic_band_pct']:>15.2f} {q_met['therapeutic_band_pct']:>15.2f}")
    print(f"{'Toxicity Violation Time (%)':<40} {c_met['toxicity_violation_pct']:>15.2f} {q_met['toxicity_violation_pct']:>15.2f}")
    print(f"{'Normal-cell Safety Violation (%)':<40} {c_met['y_safety_violation_pct']:>15.2f} {q_met['y_safety_violation_pct']:>15.2f}")
    print(f"{'Avg Toxicity':<40} {c_met['avg_toxicity']:>15.4f} {q_met['avg_toxicity']:>15.4f}")
    print(f"{'Final Toxicity':<40} {c_met['final_toxicity']:>15.4f} {q_met['final_toxicity']:>15.4f}")
    print(f"{'Final P cells':<40} {c_met['final_P']:>15.4e} {q_met['final_P']:>15.4e}")
    print(f"{'Final Q cells':<40} {c_met['final_Q']:>15.4e} {q_met['final_Q']:>15.4e}")
    print(f"{'Final Y cells':<40} {c_met['final_Y']:>15.4e} {q_met['final_Y']:>15.4e}")
    print(f"{'Avg step time (ms)':<40} {c_met['avg_step_ms']:>15.4f} {q_met['avg_step_ms']:>15.4f}")
    print(f"{'Median step time (ms)':<40} {c_met['median_step_ms']:>15.4f} {q_met['median_step_ms']:>15.4f}")
    print(f"{'Total simulation time (s)':<40} {c_met['total_time']:>15.4f} {q_met['total_time']:>15.4f}")
    print("─" * 72)

    dose_mae = np.mean(np.abs(c_res['drug_dose'] - q_res['drug_dose']))
    conc_mae = np.mean(np.abs(c_res['drug_conc'] - q_res['drug_conc']))
    speedup = q_met['avg_step_ms'] / c_met['avg_step_ms'] if c_met['avg_step_ms'] > 0 else float('inf')

    print(f"\n{'Drug Dose MAE (C vs Q)':<40} {dose_mae:>15.4f}")
    print(f"{'Drug Conc MAE (C vs Q)':<40} {conc_mae:>15.4f}")
    print(f"{'Classical speedup factor':<40} {'1.00x':>15} {f'{speedup:.1f}x slower':>15}")

    # ── Time-complexity analysis ─────────────────────────────────────
    n_input_vars = 2
    n_sets_per_var = 5
    n_rules = len(RULES)
    n_universe = len(DOSAGE_UNIVERSE)
    n_out_sets = 5

    print("\n" + "=" * 72)
    print("  TIME-COMPLEXITY ANALYSIS")
    print("=" * 72)
    print(f"""
  Let:
    V = number of input variables      = {n_input_vars}
    S = fuzzy sets per input variable   = {n_sets_per_var}
    R = number of rules                 = {n_rules}
    N = universe discretization points  = {n_universe}
    S_out = output fuzzy sets           = {n_out_sets}
    K = quantum shots (samples)         = {N_SHOTS}

  ┌─────────────────┬──────────────────────────────────┬─────────────────────────────────┐
  │ Stage           │ CLASSICAL                        │ QUANTUM                         │
  ├─────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ Fuzzification   │ O(V × S)                         │ O(V × S)  [same]                │
  │ Rule evaluation │ O(R × V)                         │ O(R)  gate placement            │
  │                 │ ({n_rules} rules × {n_input_vars} vars)              │ MCX gates; depth = {n_rules}              │
  │ Circuit build   │ —                                │ O(V×S + R)  qubits + gates      │
  │ Execution       │ —                                │ O(K × 2^Q)  simulation*         │
  │ Defuzzification │ O(S_out × N)                     │ O(S_out × N)  [same]            │
  ├─────────────────┼──────────────────────────────────┼─────────────────────────────────┤
  │ TOTAL per step  │ O(R×V + S_out×N)                 │ O(K × 2^Q + S_out×N)           │
  └─────────────────┴──────────────────────────────────┴─────────────────────────────────┘

  * Q = total qubits = V×S + S_out = {n_input_vars}×{n_sets_per_var} + {n_out_sets} = {n_input_vars * n_sets_per_var + n_out_sets} qubits.

  KEY INSIGHT FOR CANCER DRUG DELIVERY:
  ──────────────────────────────────────
  The quantum fuzzy PI controller has {n_rules} rules and {n_input_vars * n_sets_per_var + n_out_sets} qubits.
  On real quantum hardware, rule evaluation parallelism would give O(R)
  depth instead of O(R × V) classical operations. For drug scheduling
  with many rules/variables, the quantum advantage would be significant.

  On a classical SIMULATOR, quantum is slower due to O(2^Q) overhead.
""")

    # ── Plotting ─────────────────────────────────────────────────────
    time_axis = c_res['time']

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    # (0,0) Drug Concentration tracking
    ax = axes[0, 0]
    ax.plot(time_axis, c_res['drug_conc'], linewidth=2, label='Classical', color='royalblue')
    ax.plot(time_axis, q_res['drug_conc'], linewidth=2, label='Quantum', color='crimson', linestyle='--')
    ax.axhline(SP, color='green', linestyle=':', alpha=0.7, label=f'Set point = {SP}')
    ax.set_title("Drug Concentration D(t)")
    ax.set_ylabel("D (mg/ml)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Drug Dose (Control Signal)
    ax = axes[0, 1]
    ax.plot(time_axis, c_res['drug_dose'], linewidth=2, label='Classical', color='royalblue')
    ax.plot(time_axis, q_res['drug_dose'], linewidth=2, label='Quantum', color='crimson', linestyle='--')
    ax.set_title("Drug Dose u(t) — Control Signal")
    ax.set_ylabel("u (mg Day⁻¹/ml)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Toxicity
    ax = axes[1, 0]
    ax.plot(time_axis, c_res['toxicity'], linewidth=2, label='Classical', color='royalblue')
    ax.plot(time_axis, q_res['toxicity'], linewidth=2, label='Quantum', color='crimson', linestyle='--')
    ax.axhline(T_MAX, color='red', linestyle=':', alpha=0.5, label=f'Tmax = {T_MAX}')
    ax.set_title("Toxicity Level T(t)")
    ax.set_ylabel("T (mg Day⁻¹/ml)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Per-Step Inference Time
    ax = axes[1, 1]
    ax.plot(time_axis, c_res['step_times'] * 1000, linewidth=1.5, label='Classical', color='royalblue')
    ax.plot(time_axis, q_res['step_times'] * 1000, linewidth=1.5, label='Quantum', color='crimson')
    ax.set_title("Per-Step Inference Time")
    ax.set_ylabel("Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (2,0) Cancer Cells (P + Q)
    ax = axes[2, 0]
    ax.plot(time_axis, c_res['p_cells'], linewidth=2, label='P (Classical)', color='royalblue')
    ax.plot(time_axis, q_res['p_cells'], linewidth=2, label='P (Quantum)', color='crimson', linestyle='--')
    ax.plot(time_axis, c_res['q_cells'], linewidth=1.5, label='Q (Classical)', color='steelblue', linestyle='-.')
    ax.plot(time_axis, q_res['q_cells'], linewidth=1.5, label='Q (Quantum)', color='indianred', linestyle=':')
    ax.set_title("Cancer Cell Populations (P & Q)")
    ax.set_ylabel("Cell Count")
    ax.set_xlabel("Time (days)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # (2,1) Dose difference
    ax = axes[2, 1]
    diff = q_res['drug_dose'] - c_res['drug_dose']
    ax.bar(time_axis[::5], diff[::5], color='darkorange', alpha=0.7, width=0.4)
    ax.axhline(0, color='black', linestyle=':', alpha=0.4)
    ax.set_title(f"Dose Difference (Quantum − Classical)  MAE={dose_mae:.3f}")
    ax.set_ylabel("Δ Dose (mg Day⁻¹/ml)")
    ax.set_xlabel("Time (days)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Classical vs Quantum Fuzzy PI — Cancer Drug Delivery ({SP_NAME}: {SP} mg/ml)",
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig("cancer_pi_controller_comparison_results.png", dpi=150)
    print(f"\nPlot saved to cancer_pi_controller_comparison_results.png")
    if SHOW_PLOTS:
        try:
            plt.show()
        except RuntimeError as exc:
            print(f"Plot display warning (GUI backend): {exc}")
    else:
        plt.close(fig)

    # Log-scale view for cell-population analysis
    fig_log, ax_log = plt.subplots(1, 1, figsize=(10, 6))
    ax_log.plot(time_axis, np.maximum(c_res['p_cells'], 1.0), linewidth=2, label='P (Classical)', color='royalblue')
    ax_log.plot(time_axis, np.maximum(q_res['p_cells'], 1.0), linewidth=2, label='P (Quantum)', color='crimson', linestyle='--')
    ax_log.plot(time_axis, np.maximum(c_res['q_cells'], 1.0), linewidth=1.5, label='Q (Classical)', color='steelblue', linestyle='-.')
    ax_log.plot(time_axis, np.maximum(q_res['q_cells'], 1.0), linewidth=1.5, label='Q (Quantum)', color='indianred', linestyle=':')
    ax_log.set_yscale('log')
    ax_log.set_title('Cancer Cell Populations (Log Scale)')
    ax_log.set_xlabel('Time (days)')
    ax_log.set_ylabel('Cell Count (log scale)')
    ax_log.legend(fontsize=9)
    ax_log.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig("cancer_pi_controller_comparison_cells_log.png", dpi=150)
    print("Plot saved to cancer_pi_controller_comparison_cells_log.png")
    if SHOW_PLOTS:
        try:
            plt.show()
        except RuntimeError as exc:
            print(f"Plot display warning (GUI backend): {exc}")
    plt.close(fig_log)

    # ── Bar chart comparison (like paper's Fig. 24/25) ───────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    # IAE comparison
    ax = axes2[0]
    labels = ['Classical\nFuzzy PI', 'Quantum\nFuzzy PI']
    values = [c_met['IAE'], q_met['IAE']]
    bars = ax.bar(labels, values, color=['royalblue', 'crimson'], alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', fontweight='bold')
    ax.set_title("IAE Comparison")
    ax.set_ylabel("IAE Value")
    ax.grid(True, alpha=0.3, axis='y')

    # Control Signal TV
    ax = axes2[1]
    values = [c_met['TV'], q_met['TV']]
    bars = ax.bar(labels, values, color=['royalblue', 'crimson'], alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontweight='bold')
    ax.set_title("Total Variation of Control Signal")
    ax.set_ylabel("TV Value")
    ax.grid(True, alpha=0.3, axis='y')

    # Average Toxicity
    ax = axes2[2]
    values = [c_met['avg_toxicity'], q_met['avg_toxicity']]
    bars = ax.bar(labels, values, color=['royalblue', 'crimson'], alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontweight='bold')
    ax.set_title("Average Toxicity (T_avg)")
    ax.set_ylabel("Toxicity")
    ax.grid(True, alpha=0.3, axis='y')

    fig2.suptitle("Performance Metrics Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("cancer_pi_controller_comparison_bars.png", dpi=150)
    print(f"Bar chart saved to cancer_pi_controller_comparison_bars.png")
    if SHOW_PLOTS:
        try:
            plt.show()
        except RuntimeError as exc:
            print(f"Plot display warning (GUI backend): {exc}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
