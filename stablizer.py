"""
Drone Altitude Stabilization  —  Quantum Fuzzy Hover Controller
================================================================
Uses the Quantum Fuzzy Inference Engine (QFIE) to keep a drone at a
target altitude.

Inputs
------
  altitude_error : target_altitude − current_altitude   (cm)
  error_rate     : rate of change of the altitude error  (cm/s)

Output
------
  motor_thrust   : thrust adjustment relative to hover   (%)

The controller mirrors the inverted-pendulum structure from the
reference implementation, adapted for vertical drone dynamics.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from src.QFIE.FuzzyEngines import QuantumFuzzyEngine, trimf, trapmf


class DroneAltitudeController:
    """Quantum-fuzzy hover controller for a simplified drone model."""

    def __init__(self):
        # ── 1. Quantum Fuzzy Engine ──────────────────────────────────
        self.qfie = QuantumFuzzyEngine(verbose=False, encoding='linear')

        # ── 2. Universes of Discourse ────────────────────────────────
        self.altitude_error = np.linspace(-50, 50, 200)    # cm
        self.error_rate     = np.linspace(-20, 20, 200)    # cm/s
        self.motor_thrust   = np.linspace(-100, 100, 200)  # %

        self.qfie.input_variable("altitude_error", self.altitude_error)
        self.qfie.input_variable("error_rate", self.error_rate)
        self.qfie.output_variable("motor_thrust", self.motor_thrust)

        # ── 3. Membership Functions ──────────────────────────────────
        # Altitude error
        alt_sets = [
            trapmf(self.altitude_error, [-50, -50, -10, 0]),   # neg  (above target)
            trimf(self.altitude_error,  [-10,   0,  10]),      # zero (at target)
            trapmf(self.altitude_error, [  0,  10,  50, 50]),  # pos  (below target)
        ]

        # Error rate (velocity of the error)
        rate_sets = [
            trapmf(self.error_rate, [-20, -20, -5, 0]),  # neg  (error shrinking)
            trimf(self.error_rate,  [ -5,   0,  5]),     # zero (stable)
            trapmf(self.error_rate, [  0,   5, 20, 20]), # pos  (error growing)
        ]

        # Motor thrust adjustment
        thrust_sets = [
            trapmf(self.motor_thrust, [-100, -100, -50, -10]),  # neg_big (reduce)
            trimf(self.motor_thrust,  [ -20,    0,  20]),       # zero    (hold)
            trapmf(self.motor_thrust, [  10,   50, 100, 100]),  # pos_big (boost)
        ]

        self.qfie.add_input_fuzzysets(
            "altitude_error", ["neg", "zero", "pos"], alt_sets
        )
        self.qfie.add_input_fuzzysets(
            "error_rate", ["neg", "zero", "pos"], rate_sets
        )
        self.qfie.add_output_fuzzysets(
            "motor_thrust", ["neg_big", "zero", "pos_big"], thrust_sets
        )

        # ── 4. Rule Base (9 rules — full coverage) ──────────────────
        #
        # Sign convention
        #   altitude_error > 0  →  drone is below target  →  need upward thrust
        #   error_rate     > 0  →  error is growing       →  need correction
        #
        rules = [
            # ---- Balanced ----
            'if altitude_error is zero and error_rate is zero then motor_thrust is zero',

            # ---- Offset, stable ----
            'if altitude_error is pos and error_rate is zero then motor_thrust is pos_big',
            'if altitude_error is neg and error_rate is zero then motor_thrust is neg_big',

            # ---- At target but drifting ----
            'if altitude_error is zero and error_rate is pos then motor_thrust is pos_big',
            'if altitude_error is zero and error_rate is neg then motor_thrust is neg_big',

            # ---- Offset AND drifting away  →  strong correction ----
            'if altitude_error is pos and error_rate is pos then motor_thrust is pos_big',
            'if altitude_error is neg and error_rate is neg then motor_thrust is neg_big',

            # ---- Offset BUT returning  →  relax ----
            'if altitude_error is pos and error_rate is neg then motor_thrust is zero',
            'if altitude_error is neg and error_rate is pos then motor_thrust is zero',
        ]
        self.qfie.set_rules(rules)

    # ------------------------------------------------------------------ #
    # Inference helper
    # ------------------------------------------------------------------ #

    def compute_thrust(self, error, rate):
        """Return crisp thrust adjustment for given error & rate."""
        safe_error = float(np.clip(error, -50, 50))
        safe_rate  = float(np.clip(rate,  -20, 20))

        crisp_inputs = {
            'altitude_error': safe_error,
            'error_rate':     safe_rate,
        }

        self.qfie.build_inference_qc(crisp_inputs, draw_qc=False, distributed=False)
        thrust, _ = self.qfie.execute(n_shots=1024)
        return float(thrust)

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #

    def run_simulation(self, error, velocity, steps=150):
        """Simulate the drone hover control loop.

        Parameters
        ----------
        error    : float – initial altitude error (cm), positive = below target
        velocity : float – initial rate of error change (cm/s)
        steps    : int   – number of discrete time steps
        """
        errors     = []
        velocities = []
        thrusts    = []

        print(f"\n{'='*60}")
        print("  Drone Altitude Stabilization — Quantum Fuzzy Controller")
        print(f"{'='*60}")
        print(f"  Initial error : {error:+.1f} cm")
        print(f"  Initial rate  : {velocity:+.1f} cm/s")
        print(f"  Time steps    : {steps}")
        print(f"{'='*60}\n")

        dt = 0.05  # time-step size (s)

        for t in range(steps):
            thrust = self.compute_thrust(error, velocity)

            errors.append(error)
            velocities.append(velocity)
            thrusts.append(thrust)

            print(
                f"Step {t:3d}:  error={error:+8.2f} cm   "
                f"rate={velocity:+8.2f} cm/s   "
                f"thrust={thrust:+8.2f} %"
            )

            # Simplified altitude-error dynamics
            #   (mirrors the inverted-pendulum model from the reference)
            acceleration = error - thrust - 0.5 * velocity
            velocity += acceleration * dt
            error    += velocity * dt

        # ── Plotting ─────────────────────────────────────────────────
        time_axis = np.arange(steps)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Altitude error
        axes[0].plot(time_axis, errors, linewidth=2, color='royalblue')
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.4)
        axes[0].set_title("Altitude Error  (goal → 0)")
        axes[0].set_ylabel("Error (cm)")
        axes[0].grid(True, alpha=0.3)

        # Error rate
        axes[1].plot(time_axis, velocities, linewidth=2, color='darkorange')
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.4)
        axes[1].set_title("Rate of Change of Error")
        axes[1].set_ylabel("Rate (cm/s)")
        axes[1].grid(True, alpha=0.3)

        # Motor thrust
        axes[2].plot(time_axis, thrusts, linewidth=2, color='seagreen')
        axes[2].axhline(0, color='black', linestyle='--', alpha=0.4)
        axes[2].set_title("Motor Thrust Adjustment")
        axes[2].set_ylabel("Thrust (%)")
        axes[2].set_xlabel("Time Step")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            "Drone Altitude Stabilization — Quantum Fuzzy Controller",
            fontsize=14,
            fontweight='bold',
        )
        plt.tight_layout()
        plt.savefig("drone_stabilization_results.png", dpi=150)
        print(f"\nPlot saved to drone_stabilization_results.png")
        plt.show()


# ====================================================================== #
# Entry point
# ====================================================================== #

if __name__ == "__main__":
    controller = DroneAltitudeController()

    # Scenario: drone is 15 cm below target, initially stationary
    controller.run_simulation(error=-15, velocity=0, steps=150)
