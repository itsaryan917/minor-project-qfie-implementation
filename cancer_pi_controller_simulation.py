"""
Cancer Drug Delivery PI Controller — Quantum Fuzzy Simulation
==============================================================
Implements a Quantum Fuzzy PI controller for cancer chemotherapy
drug concentration regulation, based on the IFOIMC design from:

    C. Kumar et al., "IFOIMC-based FOPIPOF controller for cancer
    drug scheduling", AEUE (accepted paper).

The cancer patient model is a two-compartment model with:
  - P: Proliferating cells
  - Q: Quiescent cells
  - D: Drug concentration (controlled variable)
  - T: Toxicity level
  - Y: Normal (healthy) cell count

The QFIE replaces the classical PI controller with a quantum-fuzzy
inference engine that maps (error, error_integral) → drug_dose.

Plant transfer function (FFOPTD):
    G(s) = 3.7 * e^(-0.005s) / (3.4 * s^0.989 + 1)

Approximated as first-order with delay for simulation:
    G(s) ≈ K / (T*s + 1) * e^(-Ls)
    K = 3.7, T = 3.4, L = 0.005
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import matplotlib.pyplot as plt
from QFIE.FuzzyEngines import QuantumFuzzyEngine, trimf, trapmf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cancer Patient Model Parameters (Table 2 from paper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARAMS = {
    'j': 0.50,          # Growth rate of proliferating cells (day⁻¹)
    'k': 0.2180,        # Rate cells change from P to Q (day⁻¹)
    'l': 0.4770,        # Rate cells stop multiplying (day⁻¹)
    'x': 0.050,         # Rate Q cells change into P cells (day⁻¹)
    'sigma': 0.10,      # Speed at which normal cells grow (day⁻¹)
    'N': 1e7,           # Normal cells' holding capacity
    'P0': 2e11,         # Initial population of P cells
    'Q0': 8e11,         # Initial population of Q cells
    'Y0': 1e6,          # Initial population of normal cells
    'theta': 0.270,     # Decay of drugs (day⁻¹)
    'a': 8.40e5,        # Cell death rate per unit drug conc (day⁻¹)
    'beta': 0.40,       # Rate of removing toxins
}

# Plant model parameters (from Eq. 18)
PLANT_K = 3.7           # Gain
PLANT_T = 3.4           # Time constant (fractional approx as integer-order)
PLANT_L = 0.005         # Time delay (s) — negligible, included for completeness
PLANT_LAMBDA = 0.989    # Fractional order (≈1, so integer approx is valid)

# Controller parameters (from Section 5)
PSI = -0.0220           # Shifting parameter ψ
KP_NOMINAL = 0.9392     # Proportional gain
KI_NOMINAL = 0.2703     # Integral gain

# Set points for drug concentration (mg/ml)
SET_POINTS = {
    'S1': 12.08,
    'S2': 12.17,
    'S3': 11.66,
}

# Drug concentration bounds
D_MIN = 0.0
D_MAX = 50.0

# Toxicity bound
T_MAX = 100.0

# Simulation
SIM_DAYS = 100          # Total treatment duration (days)
DT = 0.1               # Time step (days)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cancer Patient Model (Eqs. 1–9 from paper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CancerPatientModel:
    """Two-compartment cancer patient model.

    State variables:
      P  — Proliferating cells
      Q  — Quiescent cells
      D  — Drug concentration (mg/ml)
      T  — Toxicity level (mg Day⁻¹/ml)
      Y  — Normal (healthy) cell count
    """

    def __init__(self, params=None):
        p = params or PARAMS
        self.j = p['j']
        self.k = p['k']
        self.l = p['l']
        self.x = p['x']
        self.sigma = p['sigma']
        self.N = p['N']
        self.a = p['a']
        self.theta = p['theta']
        self.beta = p['beta']

        # Initial conditions
        self.P = p['P0']
        self.Q = p['Q0']
        self.D = 0.0       # No drug initially
        self.T = 0.0       # No toxicity initially
        self.Y = p['Y0']

    def step(self, u, dt):
        """Advance one time step given drug dosage u (mg Day⁻¹/ml).

        Parameters
        ----------
        u  : float — drug administration rate (control signal)
        dt : float — time step in days

        Returns
        -------
        D  : float — current drug concentration
        """
        P, Q, D, T, Y = self.P, self.Q, self.D, self.T, self.Y

        # r(t) = a * D(t) — death rate of cancer cells per unit drug
        r = self.a * D

        # Eq (1): dP/dt = (j - k - l)*P + x*Q - r*P
        dP = (self.j - self.k - self.l) * P + self.x * Q - r * P

        # Eq (2): dQ/dt = k*P - x*Q
        dQ = self.k * P - self.x * Q

        # Eq (4): dD/dt = u - θ*D
        dD = u - self.theta * D

        # Eq (6): dT/dt = D - β*T
        dT = D - self.beta * T

        # Eq (3): dY/dt = σ*Y*(1 - Y/N) - r*Y
        dY = self.sigma * Y * (1.0 - Y / self.N) - r * Y

        # Euler integration
        self.P = max(0, P + dP * dt)
        self.Q = max(0, Q + dQ * dt)
        self.D = np.clip(D + dD * dt, D_MIN, D_MAX)
        self.T = np.clip(T + dT * dt, 0, T_MAX)
        self.Y = max(0, Y + dY * dt)

        return self.D


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quantum Fuzzy PI Controller
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QuantumFuzzyPIController:
    """Quantum Fuzzy PI controller for drug concentration regulation.

    Maps (error, integral_of_error) → drug dosage adjustment using QFIE.
    This replaces the classical IFOIMC PI^λ controller with a quantum
    fuzzy inference engine.
    """

    def __init__(self):
        self.qfie = QuantumFuzzyEngine(verbose=False, encoding='linear')

        # ── Universes of Discourse ───────────────────────────────────
        # Error: difference between desired and actual drug concentration
        self.error_universe = np.linspace(-15, 15, 200)       # mg/ml

        # Integral of error (accumulated over time)
        self.integral_universe = np.linspace(-50, 50, 200)    # mg·day/ml

        # Drug dosage output (control signal u)
        self.dosage_universe = np.linspace(0, 15, 200)        # mg Day⁻¹/ml

        self.qfie.input_variable("error", self.error_universe)
        self.qfie.input_variable("integral", self.integral_universe)
        self.qfie.output_variable("dosage", self.dosage_universe)

        # ── Membership Functions ─────────────────────────────────────
        # Error: NB (negative big), NS (negative small), ZE (zero),
        #        PS (positive small), PB (positive big)
        error_sets = [
            trapmf(self.error_universe, [-15, -15, -8, -3]),   # NB
            trimf(self.error_universe,  [-6, -3, 0]),          # NS
            trimf(self.error_universe,  [-3, 0, 3]),           # ZE
            trimf(self.error_universe,  [0, 3, 6]),            # PS
            trapmf(self.error_universe, [3, 8, 15, 15]),       # PB
        ]

        # Integral of error: NB, NS, ZE, PS, PB
        integral_sets = [
            trapmf(self.integral_universe, [-50, -50, -25, -8]),  # NB
            trimf(self.integral_universe,  [-20, -8, 0]),         # NS
            trimf(self.integral_universe,  [-8, 0, 8]),           # ZE
            trimf(self.integral_universe,  [0, 8, 20]),           # PS
            trapmf(self.integral_universe, [8, 25, 50, 50]),      # PB
        ]

        # Drug dosage: VL (very low), LO (low), ME (medium), HI (high), VH (very high)
        dosage_sets = [
            trapmf(self.dosage_universe, [0, 0, 1.5, 3]),        # VL
            trimf(self.dosage_universe,  [1.5, 3.75, 6]),        # LO
            trimf(self.dosage_universe,  [4.5, 7.5, 10.5]),      # ME
            trimf(self.dosage_universe,  [9, 11.25, 13.5]),      # HI
            trapmf(self.dosage_universe, [12, 13.5, 15, 15]),    # VH
        ]

        self.qfie.add_input_fuzzysets(
            "error", ["NB", "NS", "ZE", "PS", "PB"], error_sets
        )
        self.qfie.add_input_fuzzysets(
            "integral", ["NB", "NS", "ZE", "PS", "PB"], integral_sets
        )
        self.qfie.add_output_fuzzysets(
            "dosage", ["VL", "LO", "ME", "HI", "VH"], dosage_sets
        )

        # ── Rule Base (25 rules — full PI coverage) ─────────────────
        # Rows = error, Cols = integral
        # Convention: positive error → drug is too low → need more drug
        #
        #              integral:  NB     NS     ZE     PS     PB
        # error NB:              VL     VL     VL     LO     LO
        # error NS:              VL     LO     LO     ME     ME
        # error ZE:              LO     ME     ME     ME     HI
        # error PS:              ME     ME     HI     HI     VH
        # error PB:              HI     HI     VH     VH     VH

        rules = [
            # error NB (drug far above target → reduce dose significantly)
            'if error is NB and integral is NB then dosage is VL',
            'if error is NB and integral is NS then dosage is VL',
            'if error is NB and integral is ZE then dosage is VL',
            'if error is NB and integral is PS then dosage is LO',
            'if error is NB and integral is PB then dosage is LO',

            # error NS (drug slightly above target)
            'if error is NS and integral is NB then dosage is VL',
            'if error is NS and integral is NS then dosage is LO',
            'if error is NS and integral is ZE then dosage is LO',
            'if error is NS and integral is PS then dosage is ME',
            'if error is NS and integral is PB then dosage is ME',

            # error ZE (drug at target — maintain)
            'if error is ZE and integral is NB then dosage is LO',
            'if error is ZE and integral is NS then dosage is ME',
            'if error is ZE and integral is ZE then dosage is ME',
            'if error is ZE and integral is PS then dosage is ME',
            'if error is ZE and integral is PB then dosage is HI',

            # error PS (drug slightly below target → increase dose)
            'if error is PS and integral is NB then dosage is ME',
            'if error is PS and integral is NS then dosage is ME',
            'if error is PS and integral is ZE then dosage is HI',
            'if error is PS and integral is PS then dosage is HI',
            'if error is PS and integral is PB then dosage is VH',

            # error PB (drug far below target → increase dose strongly)
            'if error is PB and integral is NB then dosage is HI',
            'if error is PB and integral is NS then dosage is HI',
            'if error is PB and integral is ZE then dosage is VH',
            'if error is PB and integral is PS then dosage is VH',
            'if error is PB and integral is PB then dosage is VH',
        ]
        self.qfie.set_rules(rules)

        # PI state
        self.integral_error = 0.0

    def compute_dosage(self, error, dt):
        """Compute drug dosage given current error and time step.

        Parameters
        ----------
        error : float — (set_point - actual_drug_concentration)
        dt    : float — time step (days)

        Returns
        -------
        dosage : float — drug administration rate (mg Day⁻¹/ml)
        """
        # Accumulate integral
        self.integral_error += error * dt

        # Clip inputs to universe bounds
        safe_error = float(np.clip(error, -15, 15))
        safe_integral = float(np.clip(self.integral_error, -50, 50))

        crisp_inputs = {
            'error': safe_error,
            'integral': safe_integral,
        }

        self.qfie.build_inference_qc(crisp_inputs, draw_qc=False, distributed=False)
        dosage, _ = self.qfie.execute(n_shots=1024)

        # Ensure dosage is non-negative and within drug bounds (Eq. 5)
        dosage = float(np.clip(dosage, 0, 15))
        return dosage

    def reset(self):
        """Reset integral state for a new simulation."""
        self.integral_error = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_quantum_fuzzy_pi_simulation(set_point_name='S1', days=SIM_DAYS, dt=DT):
    """Run the cancer drug delivery simulation with Quantum Fuzzy PI.

    Parameters
    ----------
    set_point_name : str — 'S1', 'S2', or 'S3'
    days : float — simulation duration (days)
    dt   : float — time step (days)
    """
    set_point = SET_POINTS[set_point_name]
    steps = int(days / dt)

    print(f"\n{'='*65}")
    print("  Cancer Drug Delivery — Quantum Fuzzy PI Controller (QFIE)")
    print(f"{'='*65}")
    print(f"  Set point ({set_point_name}) : {set_point:.2f} mg/ml")
    print(f"  Duration         : {days} days")
    print(f"  Time step        : {dt} days")
    print(f"  Total steps      : {steps}")
    print(f"{'='*65}\n")

    # Initialize
    model = CancerPatientModel()
    controller = QuantumFuzzyPIController()

    # Storage
    time_axis = np.zeros(steps)
    drug_conc = np.zeros(steps)
    drug_dose = np.zeros(steps)
    toxicity = np.zeros(steps)
    p_cells = np.zeros(steps)
    q_cells = np.zeros(steps)
    y_cells = np.zeros(steps)
    errors = np.zeros(steps)

    for t in range(steps):
        time_axis[t] = t * dt

        # Current error
        error = set_point - model.D
        errors[t] = error

        # Controller output
        u = controller.compute_dosage(error, dt)

        # Advance plant
        model.step(u, dt)

        # Record state
        drug_conc[t] = model.D
        drug_dose[t] = u
        toxicity[t] = model.T
        p_cells[t] = model.P
        q_cells[t] = model.Q
        y_cells[t] = model.Y

        if t % 50 == 0:
            print(
                f"Day {time_axis[t]:6.1f}:  D={model.D:7.3f} mg/ml  "
                f"u={u:6.3f}  T={model.T:6.2f}  "
                f"P={model.P:.2e}  Q={model.Q:.2e}  Y={model.Y:.2e}"
            )

    results = {
        'time': time_axis, 'drug_conc': drug_conc, 'drug_dose': drug_dose,
        'toxicity': toxicity, 'p_cells': p_cells, 'q_cells': q_cells,
        'y_cells': y_cells, 'errors': errors, 'set_point': set_point,
        'set_point_name': set_point_name,
    }

    return results


def plot_simulation(results, save_prefix="cancer_pi_controller_quantum"):
    """Generate plots for a single simulation run."""
    t = results['time']
    sp = results['set_point']
    sp_name = results['set_point_name']

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # (0,0) Drug Concentration
    ax = axes[0, 0]
    ax.plot(t, results['drug_conc'], linewidth=2, color='royalblue')
    ax.axhline(sp, color='red', linestyle='--', alpha=0.7, label=f'Set point = {sp}')
    ax.set_title(f"Drug Concentration D(t) — {sp_name}")
    ax.set_ylabel("D (mg/ml)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (0,1) Drug Dose (Control Signal)
    ax = axes[0, 1]
    ax.plot(t, results['drug_dose'], linewidth=2, color='seagreen')
    ax.set_title("Drug Dose u(t) — Control Signal")
    ax.set_ylabel("u (mg Day⁻¹/ml)")
    ax.grid(True, alpha=0.3)

    # (1,0) Toxicity
    ax = axes[1, 0]
    ax.plot(t, results['toxicity'], linewidth=2, color='darkorange')
    ax.axhline(T_MAX, color='red', linestyle='--', alpha=0.5, label=f'Tmax = {T_MAX}')
    ax.set_title("Toxicity Level T(t)")
    ax.set_ylabel("T (mg Day⁻¹/ml)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Error
    ax = axes[1, 1]
    ax.plot(t, results['errors'], linewidth=2, color='crimson')
    ax.axhline(0, color='black', linestyle=':', alpha=0.4)
    ax.set_title("Tracking Error (Set Point − D)")
    ax.set_ylabel("Error (mg/ml)")
    ax.grid(True, alpha=0.3)

    # (2,0) Proliferating & Quiescent Cells
    ax = axes[2, 0]
    ax.plot(t, results['p_cells'], linewidth=2, label='P cells', color='crimson')
    ax.plot(t, results['q_cells'], linewidth=2, label='Q cells', color='mediumblue', linestyle='--')
    ax.set_title("Cancer Cell Populations")
    ax.set_ylabel("Cell Count")
    ax.set_xlabel("Time (days)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # (2,1) Normal Y Cells
    ax = axes[2, 1]
    ax.plot(t, results['y_cells'], linewidth=2, color='forestgreen')
    ax.set_title("Normal Cell Count Y(t)")
    ax.set_ylabel("Y cells")
    ax.set_xlabel("Time (days)")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    fig.suptitle(
        f"Cancer Drug Delivery — Quantum Fuzzy PI Controller ({sp_name}: {sp} mg/ml)",
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    fname = f"{save_prefix}_{sp_name}.png"
    plt.savefig(fname, dpi=150)
    print(f"\nPlot saved to {fname}")
    plt.show()


def main():
    """Run quantum fuzzy PI simulation for all three set points."""
    print("=" * 65)
    print("  CANCER DRUG DELIVERY — QUANTUM FUZZY PI CONTROLLER")
    print("  Based on: Kumar et al., IFOIMC FOPIPOF Controller (AEUE)")
    print("  Engine: Quantum Fuzzy Inference Engine (QFIE)")
    print("=" * 65)

    for sp_name in ['S1', 'S2', 'S3']:
        results = run_quantum_fuzzy_pi_simulation(
            set_point_name=sp_name, days=SIM_DAYS, dt=DT
        )
        plot_simulation(results)

        # Print summary metrics
        final_D = results['drug_conc'][-1]
        final_T = results['toxicity'][-1]
        final_P = results['p_cells'][-1]
        final_Q = results['q_cells'][-1]
        final_Y = results['y_cells'][-1]
        iae = np.sum(np.abs(results['errors'])) * DT

        print(f"\n  Summary for {sp_name} (set point = {SET_POINTS[sp_name]}):")
        print(f"    Final drug conc  : {final_D:.4f} mg/ml")
        print(f"    Final toxicity   : {final_T:.4f}")
        print(f"    Final P cells    : {final_P:.4e}")
        print(f"    Final Q cells    : {final_Q:.4e}")
        print(f"    Final Y cells    : {final_Y:.4e}")
        print(f"    IAE              : {iae:.4f}")
        print()


if __name__ == "__main__":
    main()
