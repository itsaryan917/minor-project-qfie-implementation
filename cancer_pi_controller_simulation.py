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

import os

import numpy as np
# Use a non-GUI backend by default on Windows/terminal runs to avoid
# Tkinter deallocation errors at interpreter shutdown.
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cancer Patient Model Parameters (Table 2 from paper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARAMS = {
    'j': 0.50,          # Growth rate of proliferating cells (day⁻¹)
    'k': 0.2180,        # Rate cells change from P to Q (day⁻¹)
    'l': 0.4770,        # Rate cells stop multiplying (day⁻¹)
    'x': 0.050,         # Rate Q cells change into P cells (day⁻¹)
    'sigma': 0.10,      # Speed at which normal cells grow (day⁻¹)
    'N': 1e9,           # Normal cells' holding capacity
    'P0': 2e11,         # Initial population of P cells
    'Q0': 8e11,         # Initial population of Q cells
    'Y0': 1e8,          # Initial population of normal cells
    'Y_MIN': 1e6,       # Lower safe bound for normal cells
    'theta': 0.270,     # Decay of drugs (day⁻¹)
    'a': 8.40e-3,       # Cell death rate per unit drug conc (day⁻¹)
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
    'S1': 12.00,
    'S2': 12.17,
    'S3': 11.66,
}

# Drug concentration bounds
D_MIN = 0.0
D_MAX = 50.0
THERAPEUTIC_D_MIN = 10.0

# Drug dosage bound (controller output)
DOSAGE_MAX = 15.0

# Toxicity bound
T_MAX = 100.0

# Normal-cell safety bound
Y_MIN_SAFE = PARAMS['Y_MIN']

# Simulation
SIM_DAYS = 100          # Total treatment duration (days)
DT = 0.1               # Time step (days)
QFIE_SHOTS = _env_int("CANCER_QFIE_SHOTS", 128)


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
        self.y_min = p.get('Y_MIN', 0.0)

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
        self.Y = np.clip(Y + dY * dt, self.y_min, self.N)

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

    def __init__(self, n_shots=QFIE_SHOTS):
        self.qfie = QuantumFuzzyEngine(verbose=False, encoding='linear')
        self.n_shots = max(1, int(n_shots))

        # ── Universes of Discourse ───────────────────────────────────
        # Error: difference between desired and actual drug concentration
        self.error_universe = np.linspace(-15, 15, 200)       # mg/ml

        # Integral of error (accumulated over time)
        self.integral_universe = np.linspace(-50, 50, 200)    # mg·day/ml

        # Fuzzy output is a dosage adjustment around the equilibrium dose.
        self.dosage_universe = np.linspace(-4, 4, 200)        # mg Day⁻¹/ml

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

        # Dosage adjustment: NB, NS, ZE, PS, PB
        dosage_sets = [
            trapmf(self.dosage_universe, [-4, -4, -2.4, -1.2]),     # NB
            trimf(self.dosage_universe,  [-2.0, -1.0, 0.0]),         # NS
            trimf(self.dosage_universe,  [-0.5, 0.0, 0.5]),          # ZE
            trimf(self.dosage_universe,  [0.0, 1.0, 2.0]),           # PS
            trapmf(self.dosage_universe, [1.2, 2.4, 4, 4]),          # PB
        ]

        self.qfie.add_input_fuzzysets(
            "error", ["NB", "NS", "ZE", "PS", "PB"], error_sets
        )
        self.qfie.add_input_fuzzysets(
            "integral", ["NB", "NS", "ZE", "PS", "PB"], integral_sets
        )
        self.qfie.add_output_fuzzysets(
            "dosage", ["NB", "NS", "ZE", "PS", "PB"], dosage_sets
        )

        # ── Rule Base (25 rules — fuzzy PI on delta-dose) ───────────
        # Rows = error, Cols = integral
        # Convention: positive error => concentration is below target => increase dose.
        #
        #              integral:  NB     NS     ZE     PS     PB
        # error NB:              NB     NB     NB     NS     ZE
        # error NS:              NB     NS     NS     ZE     PS
        # error ZE:              NB     NS     ZE     PS     PB
        # error PS:              NS     ZE     PS     PS     PB
        # error PB:              ZE     PS     PB     PB     PB

        rules = [
            # error NB (drug far above target -> decrease dose)
            'if error is NB and integral is NB then dosage is NB',
            'if error is NB and integral is NS then dosage is NB',
            'if error is NB and integral is ZE then dosage is NB',
            'if error is NB and integral is PS then dosage is NS',
            'if error is NB and integral is PB then dosage is ZE',

            # error NS (drug slightly above target)
            'if error is NS and integral is NB then dosage is NB',
            'if error is NS and integral is NS then dosage is NS',
            'if error is NS and integral is ZE then dosage is NS',
            'if error is NS and integral is PS then dosage is ZE',
            'if error is NS and integral is PB then dosage is PS',

            # error ZE (near target)
            'if error is ZE and integral is NB then dosage is NB',
            'if error is ZE and integral is NS then dosage is NS',
            'if error is ZE and integral is ZE then dosage is ZE',
            'if error is ZE and integral is PS then dosage is PS',
            'if error is ZE and integral is PB then dosage is PB',

            # error PS (drug slightly below target → increase dose)
            'if error is PS and integral is NB then dosage is NS',
            'if error is PS and integral is NS then dosage is ZE',
            'if error is PS and integral is ZE then dosage is PS',
            'if error is PS and integral is PS then dosage is PS',
            'if error is PS and integral is PB then dosage is PB',

            # error PB (drug far below target → increase dose strongly)
            'if error is PB and integral is NB then dosage is ZE',
            'if error is PB and integral is NS then dosage is PS',
            'if error is PB and integral is ZE then dosage is PB',
            'if error is PB and integral is PS then dosage is PB',
            'if error is PB and integral is PB then dosage is PB',
        ]
        self.qfie.set_rules(rules)

        # PI state
        self.integral_error = 0.0

    def compute_dosage(self, error, set_point, dt):
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
        self.integral_error = float(np.clip(self.integral_error, -50, 50))

        # Clip inputs to universe bounds
        safe_error = float(np.clip(error, -15, 15))
        safe_integral = self.integral_error

        crisp_inputs = {
            'error': safe_error,
            'integral': safe_integral,
        }

        self.qfie.build_inference_qc(crisp_inputs, draw_qc=False, distributed=False)
        delta_dose, _ = self.qfie.execute(n_shots=self.n_shots)

        # dD/dt = u - theta*D => equilibrium feedforward u_eq = theta * D_ref
        u_eq = PARAMS['theta'] * set_point
        dosage = u_eq + float(delta_dose)

        # Ensure dosage is non-negative and within drug bounds (Eq. 5)
        dosage = float(np.clip(dosage, 0, DOSAGE_MAX))
        return dosage

    def reset(self):
        """Reset integral state for a new simulation."""
        self.integral_error = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hybrid PI + Quantum Fuzzy Controller
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HybridPIFuzzyController:
    """Hybrid PI + Quantum Fuzzy controller for drug concentration regulation.

    Combines classical PI control with quantum fuzzy inference for smooth,
    adaptive dosage adjustment.
    
    Control law: u = u_eq + Kp*error + Ki*integral_error + delta_fuzzy
    where:
      u_eq = theta * set_point  (equilibrium feedforward)
      Kp*error  (proportional term)
      Ki*integral_error  (integral term)
      delta_fuzzy  (from QFIE: error, integral_error → [-4,4] adjustment)
    """

    def __init__(self, Kp=0.1, Ki=0.05, n_shots=QFIE_SHOTS):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.qfie = QuantumFuzzyEngine(verbose=False, encoding='linear')
        self.n_shots = max(1, int(n_shots))

        # ── Universes of Discourse ───────────────────────────────────
        self.error_universe = np.linspace(-15, 15, 200)       # mg/ml
        self.integral_universe = np.linspace(-50, 50, 200)    # mg·day/ml
        self.dosage_universe = np.linspace(-4, 4, 200)        # mg Day⁻¹/ml

        self.qfie.input_variable("error", self.error_universe)
        self.qfie.input_variable("integral", self.integral_universe)
        self.qfie.output_variable("dosage", self.dosage_universe)

        # ── Membership Functions ─────────────────────────────────────
        error_sets = [
            trapmf(self.error_universe, [-15, -15, -8, -3]),   # NB
            trimf(self.error_universe,  [-6, -3, 0]),          # NS
            trimf(self.error_universe,  [-3, 0, 3]),           # ZE
            trimf(self.error_universe,  [0, 3, 6]),            # PS
            trapmf(self.error_universe, [3, 8, 15, 15]),       # PB
        ]

        integral_sets = [
            trapmf(self.integral_universe, [-50, -50, -25, -8]),  # NB
            trimf(self.integral_universe,  [-20, -8, 0]),         # NS
            trimf(self.integral_universe,  [-8, 0, 8]),           # ZE
            trimf(self.integral_universe,  [0, 8, 20]),           # PS
            trapmf(self.integral_universe, [8, 25, 50, 50]),      # PB
        ]

        dosage_sets = [
            trapmf(self.dosage_universe, [-4, -4, -2.4, -1.2]),     # NB
            trimf(self.dosage_universe,  [-2.0, -1.0, 0.0]),         # NS
            trimf(self.dosage_universe,  [-0.5, 0.0, 0.5]),          # ZE
            trimf(self.dosage_universe,  [0.0, 1.0, 2.0]),           # PS
            trapmf(self.dosage_universe, [1.2, 2.4, 4, 4]),          # PB
        ]

        self.qfie.add_input_fuzzysets(
            "error", ["NB", "NS", "ZE", "PS", "PB"], error_sets
        )
        self.qfie.add_input_fuzzysets(
            "integral", ["NB", "NS", "ZE", "PS", "PB"], integral_sets
        )
        self.qfie.add_output_fuzzysets(
            "dosage", ["NB", "NS", "ZE", "PS", "PB"], dosage_sets
        )

        # ── Rule Base (25 rules) ──────────────────────────────────────
        rules = [
            'if error is NB and integral is NB then dosage is NB',
            'if error is NB and integral is NS then dosage is NB',
            'if error is NB and integral is ZE then dosage is NB',
            'if error is NB and integral is PS then dosage is NS',
            'if error is NB and integral is PB then dosage is ZE',
            'if error is NS and integral is NB then dosage is NB',
            'if error is NS and integral is NS then dosage is NS',
            'if error is NS and integral is ZE then dosage is NS',
            'if error is NS and integral is PS then dosage is ZE',
            'if error is NS and integral is PB then dosage is PS',
            'if error is ZE and integral is NB then dosage is NB',
            'if error is ZE and integral is NS then dosage is NS',
            'if error is ZE and integral is ZE then dosage is ZE',
            'if error is ZE and integral is PS then dosage is PS',
            'if error is ZE and integral is PB then dosage is PB',
            'if error is PS and integral is NB then dosage is NS',
            'if error is PS and integral is NS then dosage is ZE',
            'if error is PS and integral is ZE then dosage is PS',
            'if error is PS and integral is PS then dosage is PS',
            'if error is PS and integral is PB then dosage is PB',
            'if error is PB and integral is NB then dosage is ZE',
            'if error is PB and integral is NS then dosage is PS',
            'if error is PB and integral is ZE then dosage is PB',
            'if error is PB and integral is PS then dosage is PB',
            'if error is PB and integral is PB then dosage is PB',
        ]
        self.qfie.set_rules(rules)

        self.integral_error = 0.0

    def compute_dosage(self, error, set_point, dt):
        """Compute hybrid dosage: u = u_eq + Kp*e + Ki*∫e + delta_fuzzy
        
        Parameters
        ----------
        error : float — (set_point - actual_drug_concentration)
        set_point : float — target drug concentration
        dt : float — time step (days)

        Returns
        -------
        dosage : float — drug administration rate (mg Day⁻¹/ml)
        """
        # Accumulate integral with anti-windup
        self.integral_error += error * dt
        self.integral_error = float(np.clip(self.integral_error, -50, 50))

        # Classical PI contributions
        pi_contribution = self.Kp * error + self.Ki * self.integral_error

        # Fuzzy adjustment via QFIE
        safe_error = float(np.clip(error, -15, 15))
        crisp_inputs = {'error': safe_error, 'integral': self.integral_error}

        self.qfie.build_inference_qc(crisp_inputs, draw_qc=False, distributed=False)
        delta_fuzzy, _ = self.qfie.execute(n_shots=self.n_shots)
        delta_fuzzy = float(delta_fuzzy)

        # Equilibrium feedforward
        u_eq = PARAMS['theta'] * set_point

        # Combined control signal
        dosage = u_eq + pi_contribution + delta_fuzzy
        dosage = float(np.clip(dosage, 0, DOSAGE_MAX))
        return dosage

    def reset(self):
        """Reset integral state for a new simulation."""
        self.integral_error = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared Metrics & Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(results, dt=None):
    """Compute fairness metrics from a simulation result dict.
    
    Parameters
    ----------
    results : dict — output from run_quantum_fuzzy_pi_simulation or run_hybrid_pi_fuzzy_simulation
    dt : float — time step; if None, inferred from results['time']
    
    Returns
    -------
    metrics : dict with tracking accuracy, therapeutic compliance, safety metrics
    """
    if dt is None:
        dt = results['time'][1] - results['time'][0] if len(results['time']) > 1 else DT
    
    metrics = {
        # Tracking accuracy
        'iae': np.sum(np.abs(results['errors'])) * dt,  # Integral Absolute Error
        'ise': np.sum(results['errors']**2) * dt,       # Integral Squared Error
        'final_error': abs(results['errors'][-1]),
        'steady_state_error': np.mean(results['errors'][-int(50/dt):]) if int(50/dt) > 0 else np.mean(results['errors']),
        
        # Therapeutic compliance
        'therapeutic_band_pct': 100 * np.mean(results['in_therapeutic_band']),
        'time_in_band_days': np.sum(results['in_therapeutic_band']) * dt,
        
        # Safety constraints
        'max_toxicity': np.max(results['toxicity']),
        'toxicity_violation_pct': 100 * np.mean(results['toxicity_violation']),
        'min_y_cells': np.min(results['y_cells']),
        'y_safety_violation_pct': 100 * np.mean(results['y_safety_violation']),
        
        # Control signal activity
        'mean_dosage': np.mean(results['drug_dose']),
        'max_dosage': np.max(results['drug_dose']),
        'dosage_variance': np.var(results['drug_dose']),
        'dosage_changes': np.sum(np.abs(np.diff(results['drug_dose']))) * dt,
        
        # Final state
        'final_drug_conc': results['drug_conc'][-1],
        'final_p_cells': results['p_cells'][-1],
        'final_q_cells': results['q_cells'][-1],
        'final_y_cells': results['y_cells'][-1],
    }
    return metrics


def run_quantum_fuzzy_pi_simulation(set_point_name='S1', days=SIM_DAYS, dt=DT):
    """Run the cancer drug delivery simulation with Quantum Fuzzy PI.

    Parameters
    ----------
    set_point_name : str — 'S1', 'S2', or 'S3'
    days : float — simulation duration (days)
    dt   : float — time step (days)
    """
    set_point = SET_POINTS[set_point_name]
    if not (THERAPEUTIC_D_MIN <= set_point <= D_MAX):
        raise ValueError(
            f"Set point {set_point:.2f} mg/ml must be in "
            f"[{THERAPEUTIC_D_MIN:.2f}, {D_MAX:.2f}] mg/ml."
        )

    steps = int(days / dt)

    print(f"\n{'='*65}")
    print("  Cancer Drug Delivery — Quantum Fuzzy PI Controller (QFIE)")
    print(f"{'='*65}")
    print(f"  Set point ({set_point_name}) : {set_point:.2f} mg/ml")
    print(f"  Duration         : {days} days")
    print(f"  Time step        : {dt} days")
    print(f"  Total steps      : {steps}")
    print(f"  Quantum shots    : {QFIE_SHOTS}")
    print(f"{'='*65}\n")

    # Initialize
    model = CancerPatientModel()
    controller = QuantumFuzzyPIController(n_shots=QFIE_SHOTS)

    # Storage
    time_axis = np.zeros(steps)
    drug_conc = np.zeros(steps)
    drug_dose = np.zeros(steps)
    toxicity = np.zeros(steps)
    p_cells = np.zeros(steps)
    q_cells = np.zeros(steps)
    y_cells = np.zeros(steps)
    errors = np.zeros(steps)
    in_therapeutic_band = np.zeros(steps, dtype=bool)
    toxicity_violation = np.zeros(steps, dtype=bool)
    y_safety_violation = np.zeros(steps, dtype=bool)

    for t in range(steps):
        time_axis[t] = t * dt

        # Current error
        error = set_point - model.D
        errors[t] = error

        # Controller output
        u = controller.compute_dosage(error, set_point, dt)

        # Advance plant
        model.step(u, dt)

        # Record state
        drug_conc[t] = model.D
        drug_dose[t] = u
        toxicity[t] = model.T
        p_cells[t] = model.P
        q_cells[t] = model.Q
        y_cells[t] = model.Y
        in_therapeutic_band[t] = THERAPEUTIC_D_MIN <= model.D <= D_MAX
        toxicity_violation[t] = model.T > T_MAX
        y_safety_violation[t] = model.Y < Y_MIN_SAFE

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
        'in_therapeutic_band': in_therapeutic_band,
        'toxicity_violation': toxicity_violation,
        'y_safety_violation': y_safety_violation,
        'set_point_name': set_point_name,
        'controller_type': 'fuzzy',
    }

    return results


def run_hybrid_pi_fuzzy_simulation(set_point_name='S1', days=SIM_DAYS, dt=DT, Kp=0.1, Ki=0.05):
    """Run cancer drug delivery simulation with Hybrid PI + Quantum Fuzzy controller.
    
    Parameters
    ----------
    set_point_name : str — 'S1', 'S2', or 'S3'
    days : float — simulation duration (days)
    dt : float — time step (days)
    Kp : float — proportional gain (classical PI term)
    Ki : float — integral gain (classical PI term)
    
    Returns
    -------
    results : dict — identical structure to run_quantum_fuzzy_pi_simulation
    """
    set_point = SET_POINTS[set_point_name]
    if not (THERAPEUTIC_D_MIN <= set_point <= D_MAX):
        raise ValueError(
            f"Set point {set_point:.2f} mg/ml must be in "
            f"[{THERAPEUTIC_D_MIN:.2f}, {D_MAX:.2f}] mg/ml."
        )

    steps = int(days / dt)

    print(f"\n{'='*70}")
    print("  Cancer Drug Delivery — HYBRID PI + Quantum Fuzzy Controller (QFIE)")
    print(f"{'='*70}")
    print(f"  Set point ({set_point_name}) : {set_point:.2f} mg/ml")
    print(f"  Kp (proportional gain) : {Kp:.4f}")
    print(f"  Ki (integral gain)     : {Ki:.4f}")
    print(f"  Duration         : {days} days")
    print(f"  Time step        : {dt} days")
    print(f"  Total steps      : {steps}")
    print(f"  Quantum shots    : {QFIE_SHOTS}")
    print(f"{'='*70}\n")

    # Initialize
    model = CancerPatientModel()
    controller = HybridPIFuzzyController(Kp=Kp, Ki=Ki, n_shots=QFIE_SHOTS)

    # Storage
    time_axis = np.zeros(steps)
    drug_conc = np.zeros(steps)
    drug_dose = np.zeros(steps)
    toxicity = np.zeros(steps)
    p_cells = np.zeros(steps)
    q_cells = np.zeros(steps)
    y_cells = np.zeros(steps)
    errors = np.zeros(steps)
    in_therapeutic_band = np.zeros(steps, dtype=bool)
    toxicity_violation = np.zeros(steps, dtype=bool)
    y_safety_violation = np.zeros(steps, dtype=bool)

    for t in range(steps):
        time_axis[t] = t * dt

        # Current error
        error = set_point - model.D
        errors[t] = error

        # Controller output
        u = controller.compute_dosage(error, set_point, dt)

        # Advance plant
        model.step(u, dt)

        # Record state
        drug_conc[t] = model.D
        drug_dose[t] = u
        toxicity[t] = model.T
        p_cells[t] = model.P
        q_cells[t] = model.Q
        y_cells[t] = model.Y
        in_therapeutic_band[t] = THERAPEUTIC_D_MIN <= model.D <= D_MAX
        toxicity_violation[t] = model.T > T_MAX
        y_safety_violation[t] = model.Y < Y_MIN_SAFE

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
        'in_therapeutic_band': in_therapeutic_band,
        'toxicity_violation': toxicity_violation,
        'y_safety_violation': y_safety_violation,
        'set_point_name': set_point_name,
        'controller_type': 'hybrid',
        'Kp': Kp, 'Ki': Ki,
    }

    return results


def compare_fuzzy_vs_hybrid(set_point_name='S1', Kp=0.1, Ki=0.05):
    """Run both controllers on identical scenario and compare metrics.
    
    Parameters
    ----------
    set_point_name : str — 'S1', 'S2', or 'S3'
    Kp : float — proportional gain for hybrid mode
    Ki : float — integral gain for hybrid mode
    
    Returns
    -------
    comparison : dict with keys:
      - 'fuzzy': results dict from fuzzy-only
      - 'hybrid': results dict from hybrid
      - 'metrics_fuzzy': computed metrics for fuzzy
      - 'metrics_hybrid': computed metrics for hybrid
      - 'delta_metrics': differential metrics
    """
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Fuzzy-Only vs Hybrid PI+Fuzzy ({set_point_name})")
    print(f"{'='*70}\n")
    
    fuzzy_results = run_quantum_fuzzy_pi_simulation(set_point_name=set_point_name)
    hybrid_results = run_hybrid_pi_fuzzy_simulation(
        set_point_name=set_point_name, Kp=Kp, Ki=Ki
    )
    
    fuzzy_metrics = compute_metrics(fuzzy_results)
    hybrid_metrics = compute_metrics(hybrid_results)
    
    # Compute differential metrics
    mae_conc = np.mean(np.abs(
        fuzzy_results['drug_conc'] - hybrid_results['drug_conc']
    ))
    mae_dose = np.mean(np.abs(
        fuzzy_results['drug_dose'] - hybrid_results['drug_dose']
    ))
    
    iae_fuzzy = fuzzy_metrics['iae']
    iae_hybrid = hybrid_metrics['iae']
    iae_improvement_pct = 100 * (iae_fuzzy - iae_hybrid) / iae_fuzzy if iae_fuzzy > 0 else 0
    
    delta_metrics = {
        'mae_drug_conc': mae_conc,
        'mae_drug_dose': mae_dose,
        'iae_fuzzy': iae_fuzzy,
        'iae_hybrid': iae_hybrid,
        'iae_improvement_pct': iae_improvement_pct,
        'therapeutic_diff_pct': hybrid_metrics['therapeutic_band_pct'] - fuzzy_metrics['therapeutic_band_pct'],
        'toxicity_diff_pct': hybrid_metrics['toxicity_violation_pct'] - fuzzy_metrics['toxicity_violation_pct'],
        'y_safety_diff_pct': hybrid_metrics['y_safety_violation_pct'] - fuzzy_metrics['y_safety_violation_pct'],
    }
    
    return {
        'fuzzy': fuzzy_results,
        'hybrid': hybrid_results,
        'metrics_fuzzy': fuzzy_metrics,
        'metrics_hybrid': hybrid_metrics,
        'delta_metrics': delta_metrics,
    }


def plot_simulation(results, save_prefix="cancer_pi_controller_quantum"):
    """Generate linear and log-scale plots for a single simulation run."""
    t = results['time']
    sp = results['set_point']
    sp_name = results['set_point_name']

    def _make_figure(log_cells=False):
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
        p_vals = np.maximum(results['p_cells'], 1.0) if log_cells else results['p_cells']
        q_vals = np.maximum(results['q_cells'], 1.0) if log_cells else results['q_cells']
        ax.plot(t, p_vals, linewidth=2, label='P cells', color='crimson')
        ax.plot(t, q_vals, linewidth=2, label='Q cells', color='mediumblue', linestyle='--')
        ax.set_title("Cancer Cell Populations")
        ax.set_ylabel("Cell Count")
        ax.set_xlabel("Time (days)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if log_cells:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # (2,1) Normal Y Cells
        ax = axes[2, 1]
        y_vals = np.maximum(results['y_cells'], 1.0) if log_cells else results['y_cells']
        ax.plot(t, y_vals, linewidth=2, color='forestgreen')
        ax.set_title("Normal Cell Count Y(t)")
        ax.set_ylabel("Y cells")
        ax.set_xlabel("Time (days)")
        ax.grid(True, alpha=0.3)
        if log_cells:
            ax.set_yscale('log')
        else:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # Build title based on controller type
        scale_label = "log-cell" if log_cells else "linear"
        ctrl_type = results.get('controller_type', 'unknown')
        if ctrl_type == 'hybrid':
            kp = results.get('Kp', 0)
            ki = results.get('Ki', 0)
            title = f"Cancer Drug Delivery — Hybrid PI+Fuzzy Controller (Kp={kp:.3f}, Ki={ki:.3f}, {sp_name}: {sp} mg/ml, {scale_label})"
        else:
            title = f"Cancer Drug Delivery — Quantum Fuzzy PI Controller ({sp_name}: {sp} mg/ml, {scale_label})"
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Include controller type in filename
        ctrl_type = results.get('controller_type', 'unknown')
        if ctrl_type == 'hybrid':
            kp = results.get('Kp', 0)
            ki = results.get('Ki', 0)
            out_name = f"{save_prefix}_hybrid_Kp{kp:.3f}_Ki{ki:.3f}_{sp_name}_{scale_label}.png"
        else:
            out_name = f"{save_prefix}_{sp_name}_{scale_label}.png"
        
        plt.savefig(out_name, dpi=150)
        print(f"\nPlot saved to {out_name}")
        return fig

    fig_linear = _make_figure(log_cells=False)
    fig_log = _make_figure(log_cells=True)

    if SHOW_PLOTS:
        try:
            plt.show()
        except RuntimeError as exc:
            print(f"Plot display warning (GUI backend): {exc}")

    plt.close(fig_linear)
    plt.close(fig_log)


def _print_summary(results, metrics):
    """Print summary metrics for a simulation."""
    sp_name = results['set_point_name']
    final_D = results['drug_conc'][-1]
    final_T = results['toxicity'][-1]
    final_P = results['p_cells'][-1]
    final_Q = results['q_cells'][-1]
    final_Y = results['y_cells'][-1]
    
    iae = metrics['iae']
    therapeutic_pct = metrics['therapeutic_band_pct']
    toxicity_viol_pct = metrics['toxicity_violation_pct']
    y_viol_pct = metrics['y_safety_violation_pct']

    print(f"\n  Summary for {sp_name} (set point = {SET_POINTS[sp_name]}):")
    print(f"    Final drug conc  : {final_D:.4f} mg/ml")
    print(f"    Final toxicity   : {final_T:.4f}")
    print(f"    Final P cells    : {final_P:.4e}")
    print(f"    Final Q cells    : {final_Q:.4e}")
    print(f"    Final Y cells    : {final_Y:.4e}")
    print(f"    IAE              : {iae:.4f}")
    print(f"    In therapeutic band (10-50 mg/ml): {therapeutic_pct:6.2f}%")
    print(f"    Toxicity violation time (T > {T_MAX:.0f}) : {toxicity_viol_pct:6.2f}%")
    print(f"    Normal-cell safety violation (Y < {Y_MIN_SAFE:.1e}): {y_viol_pct:6.2f}%")
    print()


def _print_comparison_table(set_point_name, comparison_data):
    """Print a formatted comparison table for fuzzy vs hybrid."""
    print(f"\n{'='*95}")
    print(f"  COMPARISON TABLE: {set_point_name}")
    print(f"{'='*95}")
    print(f"{'Metric':<40} {'Fuzzy-Only':<20} {'Hybrid PI+Fuzzy':<20}")
    print(f"{'-'*95}")
    
    fuzzy_m = comparison_data['metrics_fuzzy']
    hybrid_m = comparison_data['metrics_hybrid']
    delta = comparison_data['delta_metrics']
    
    metrics_to_show = [
        ('IAE (Integral Absolute Error)', 'iae'),
        ('ISE (Integral Squared Error)', 'ise'),
        ('Final Error (mg/ml)', 'final_error'),
        ('Steady-State Error (mg/ml)', 'steady_state_error'),
        ('Therapeutic Band Time (%)', 'therapeutic_band_pct'),
        ('Toxicity Violations (%)', 'toxicity_violation_pct'),
        ('Safety Violations (Y) (%)', 'y_safety_violation_pct'),
        ('Mean Dosage (mg/ml)', 'mean_dosage'),
        ('Max Dosage (mg/ml)', 'max_dosage'),
        ('Dosage Variance', 'dosage_variance'),
        ('Final Drug Conc (mg/ml)', 'final_drug_conc'),
        ('Final Y Cells', 'final_y_cells'),
    ]
    
    for label, key in metrics_to_show:
        fuzzy_val = fuzzy_m.get(key, 0)
        hybrid_val = hybrid_m.get(key, 0)
        print(f"{label:<40} {fuzzy_val:>18.4f}  {hybrid_val:>18.4f}")
    
    print(f"{'-'*95}")
    print(f"{'Delta (Hybrid - Fuzzy):':<40} {'Absolute':<20} {'Percent':<20}")
    print(f"{'-'*95}")
    iae_delta = delta['iae_fuzzy'] - delta['iae_hybrid']
    print(f"{'IAE Improvement':<40} {iae_delta:>18.4f}  {delta['iae_improvement_pct']:>18.2f}%")
    print(f"{'Therapeutic Band Diff':<40} {delta['therapeutic_diff_pct']:>18.2f}%")
    print(f"{'Toxicity Violation Diff':<40} {delta['toxicity_diff_pct']:>18.2f}%")
    print(f"{'Y-Safety Violation Diff':<40} {delta['y_safety_diff_pct']:>18.2f}%")
    print(f"{'-'*95}")


def main(mode='fuzzy-only', Kp=0.1, Ki=0.05):
    """Run cancer drug delivery PI simulation for all three set points.
    
    Parameters:
    -----------
    mode : str
        'fuzzy-only' - Quantum Fuzzy PI Controller only
        'hybrid' - Hybrid PI + Quantum Fuzzy Controller only
        'both' - Run both sequentially
    Kp : float
        Proportional gain for hybrid controller
    Ki : float
        Integral gain for hybrid controller
    """
    mode = mode.lower().strip()
    
    if mode in ('fuzzy-only', 'fuzzy'):
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
            therapeutic_pct = 100 * np.mean(results['in_therapeutic_band'])
            toxicity_viol_pct = 100 * np.mean(results['toxicity_violation'])
            y_viol_pct = 100 * np.mean(results['y_safety_violation'])

            print(f"\n  Summary for {sp_name} (set point = {SET_POINTS[sp_name]}):")
            print(f"    Final drug conc  : {final_D:.4f} mg/ml")
            print(f"    Final toxicity   : {final_T:.4f}")
            print(f"    Final P cells    : {final_P:.4e}")
            print(f"    Final Q cells    : {final_Q:.4e}")
            print(f"    Final Y cells    : {final_Y:.4e}")
            print(f"    IAE              : {iae:.4f}")
            print(f"    In therapeutic band (10-50 mg/ml): {therapeutic_pct:6.2f}%")
            print(f"    Toxicity violation time (T > {T_MAX:.0f}) : {toxicity_viol_pct:6.2f}%")
            print(f"    Normal-cell safety violation (Y < {Y_MIN_SAFE:.1e}): {y_viol_pct:6.2f}%")
            print()
    
    elif mode == 'hybrid':
        print("=" * 65)
        print("  CANCER DRUG DELIVERY — HYBRID PI + QUANTUM FUZZY CONTROLLER")
        print("  Based on: Kumar et al., IFOIMC FOPIPOF Controller (AEUE)")
        print("  Engine: Quantum Fuzzy Inference Engine (QFIE)")
        print(f"  Parameters: Kp={Kp}, Ki={Ki}")
        print("=" * 65)

        for sp_name in ['S1', 'S2', 'S3']:
            results = run_hybrid_pi_fuzzy_simulation(
                set_point_name=sp_name, days=SIM_DAYS, dt=DT, Kp=Kp, Ki=Ki
            )
            plot_simulation(results)

            # Print summary metrics
            final_D = results['drug_conc'][-1]
            final_T = results['toxicity'][-1]
            final_P = results['p_cells'][-1]
            final_Q = results['q_cells'][-1]
            final_Y = results['y_cells'][-1]
            iae = np.sum(np.abs(results['errors'])) * DT
            therapeutic_pct = 100 * np.mean(results['in_therapeutic_band'])
            toxicity_viol_pct = 100 * np.mean(results['toxicity_violation'])
            y_viol_pct = 100 * np.mean(results['y_safety_violation'])

            print(f"\n  Summary for {sp_name} (set point = {SET_POINTS[sp_name]}):")
            print(f"    Final drug conc  : {final_D:.4f} mg/ml")
            print(f"    Final toxicity   : {final_T:.4f}")
            print(f"    Final P cells    : {final_P:.4e}")
            print(f"    Final Q cells    : {final_Q:.4e}")
            print(f"    Final Y cells    : {final_Y:.4e}")
            print(f"    IAE              : {iae:.4f}")
            print(f"    In therapeutic band (10-50 mg/ml): {therapeutic_pct:6.2f}%")
            print(f"    Toxicity violation time (T > {T_MAX:.0f}) : {toxicity_viol_pct:6.2f}%")
            print(f"    Normal-cell safety violation (Y < {Y_MIN_SAFE:.1e}): {y_viol_pct:6.2f}%")
            print()
    
    elif mode == 'both':
        # Run both modes
        main(mode='fuzzy-only')
        print("\n\n")
        main(mode='hybrid', Kp=Kp, Ki=Ki)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: 'fuzzy-only', 'hybrid', or 'both'")


if __name__ == "__main__":
    main()
