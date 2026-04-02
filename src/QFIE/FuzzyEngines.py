"""
Quantum Fuzzy Inference Engine (QFIE)
=====================================
Provides QuantumFuzzyEngine — a high-level class that:
  1. Accepts fuzzy variable definitions and membership functions
  2. Fuzzifies crisp inputs
  3. Encodes membership degrees as qubit rotation angles
  4. Applies fuzzy rules via multi-controlled-X gates
  5. Executes on a quantum simulator and defuzzifies the result
"""

import math
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# ---------------------------------------------------------------------------
# Membership-function helpers (so we don't need scikit-fuzzy)
# ---------------------------------------------------------------------------

def trimf(x, abc):
    """Triangular membership function.

    Parameters
    ----------
    x : np.ndarray  – universe of discourse
    abc : (a, b, c) – feet and peak of the triangle
    """
    a, b, c = [float(v) for v in abc]
    y = np.zeros_like(x, dtype=float)

    # Rising slope
    if b > a:
        idx = (a < x) & (x <= b)
        y[idx] = (x[idx] - a) / (b - a)

    # Falling slope
    if c > b:
        idx = (b < x) & (x < c)
        y[idx] = (c - x[idx]) / (c - b)

    # Peak
    y[x == b] = 1.0
    return y


def trapmf(x, abcd):
    """Trapezoidal membership function.

    Parameters
    ----------
    x : np.ndarray    – universe of discourse
    abcd : (a, b, c, d) – four vertices of the trapezoid
    """
    a, b, c, d = [float(v) for v in abcd]
    y = np.zeros_like(x, dtype=float)

    # Rising slope
    if b > a:
        idx = (a < x) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)

    # Flat top
    idx = (b <= x) & (x <= c)
    y[idx] = 1.0

    # Falling slope
    if d > c:
        idx = (c < x) & (x < d)
        y[idx] = (d - x[idx]) / (d - c)

    return y


# ---------------------------------------------------------------------------
# QuantumFuzzyEngine
# ---------------------------------------------------------------------------

class QuantumFuzzyEngine:
    """High-level Quantum Fuzzy Inference Engine.

    Usage
    -----
    >>> engine = QuantumFuzzyEngine(verbose=False, encoding='linear')
    >>> engine.input_variable("x", np.linspace(-10, 10, 200))
    >>> engine.output_variable("y", np.linspace(-50, 50, 200))
    >>> engine.add_input_fuzzysets("x", ["neg", "zero", "pos"], [mf1, mf2, mf3])
    >>> engine.add_output_fuzzysets("y", ["low", "mid", "high"], [mf4, mf5, mf6])
    >>> engine.set_rules([...])
    >>> engine.build_inference_qc({"x": 3.5}, draw_qc=False)
    >>> crisp_out, strengths = engine.execute(n_shots=1024)
    """

    def __init__(self, verbose=False, encoding='linear'):
        self.verbose = verbose
        self.encoding = encoding          # 'linear' (one qubit per set)

        # Variable stores
        self.input_vars = {}              # name  -> universe (ndarray)
        self.output_vars = {}             # name  -> universe (ndarray)
        self.input_fuzzysets = {}         # name  -> {set_names, mfs}
        self.output_fuzzysets = {}        # name  -> {set_names, mfs}
        self.rules = []

        # Internal circuit state (rebuilt each inference call)
        self._qc = None
        self._input_regs = {}
        self._output_reg = None
        self._output_cr = None
        self._sampler = None

    # ------------------------------------------------------------------ #
    # Variable & fuzzy-set registration
    # ------------------------------------------------------------------ #

    def input_variable(self, name, universe):
        """Register an input linguistic variable."""
        self.input_vars[name] = np.asarray(universe, dtype=float)

    def output_variable(self, name, universe):
        """Register an output linguistic variable."""
        self.output_vars[name] = np.asarray(universe, dtype=float)

    def add_input_fuzzysets(self, var_name, set_names, membership_fns):
        """Attach named fuzzy sets (with pre-computed MF arrays) to an input."""
        self.input_fuzzysets[var_name] = {
            'set_names': list(set_names),
            'mfs': list(membership_fns),
        }

    def add_output_fuzzysets(self, var_name, set_names, membership_fns):
        """Attach named fuzzy sets to an output."""
        self.output_fuzzysets[var_name] = {
            'set_names': list(set_names),
            'mfs': list(membership_fns),
        }

    def set_rules(self, rules):
        """Set the fuzzy rule base (list of rule strings)."""
        self.rules = list(rules)

    # ------------------------------------------------------------------ #
    # Fuzzification
    # ------------------------------------------------------------------ #

    def _fuzzify(self, crisp_inputs):
        """Return {var_name: {set_name: degree, ...}, ...}."""
        membership = {}
        for var_name, value in crisp_inputs.items():
            universe = self.input_vars[var_name]
            info = self.input_fuzzysets[var_name]
            degrees = {}
            for sname, mf in zip(info['set_names'], info['mfs']):
                degree = float(np.interp(value, universe, mf))
                degree = max(0.0, min(1.0, degree))
                degrees[sname] = degree
            membership[var_name] = degrees
        return membership

    # ------------------------------------------------------------------ #
    # Rule parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_rule(rule_str, input_vars):
        """Parse a natural-language rule string.

        Expects the form:
            'if <var> is <set> [and <var> is <set> ...] then <var> is <set>'

        Returns
        -------
        antecedents : list of (var_name, set_name)
        consequent  : (var_name, set_name)
        """
        tokens = rule_str.strip().split()
        antecedents = []
        consequent = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ('if', 'and'):
                i += 1
                continue
            if tok == 'then':
                # consequent follows: var is set
                consequent = (tokens[i + 1], tokens[i + 3])
                break
            if tok in input_vars:
                # var is set_name
                antecedents.append((tok, tokens[i + 2]))
                i += 3
                continue
            i += 1
        return antecedents, consequent

    # ------------------------------------------------------------------ #
    # Quantum-circuit construction
    # ------------------------------------------------------------------ #

    def build_inference_qc(self, crisp_inputs, draw_qc=False, distributed=False):
        """Build the quantum inference circuit for the given crisp inputs.

        Steps:
          1. Fuzzify each input.
          2. Create one qubit per fuzzy set (linear encoding).
          3. Apply Ry(θ) rotations so that P(|1⟩) = membership degree.
          4. Apply each rule as an MCX (multi-controlled X) gate.
          5. Attach measurement on output qubits.
        """
        membership = self._fuzzify(crisp_inputs)

        if self.verbose:
            print("Membership degrees:")
            for var, degs in membership.items():
                print(f"  {var}: {degs}")

        qc = QuantumCircuit()
        self._input_regs = {}

        # --- Input registers (one qubit per fuzzy set) ---
        for var_name in self.input_vars:
            n_sets = len(self.input_fuzzysets[var_name]['set_names'])
            qr = QuantumRegister(n_sets, name=var_name)
            qc.add_register(qr)
            self._input_regs[var_name] = qr

        # --- Output register ---
        out_var = list(self.output_vars.keys())[0]
        n_out = len(self.output_fuzzysets[out_var]['set_names'])
        self._output_reg = QuantumRegister(n_out, name=out_var)
        qc.add_register(self._output_reg)

        # --- Encode membership degrees via Ry rotations ---
        # P(|1⟩) = sin²(θ/2) = μ  ⟹  θ = 2·arcsin(√μ)
        for var_name, degrees in membership.items():
            set_names = self.input_fuzzysets[var_name]['set_names']
            qr = self._input_regs[var_name]
            for i, sname in enumerate(set_names):
                mu = degrees[sname]
                if mu > 1e-12:
                    theta = 2.0 * math.asin(math.sqrt(min(mu, 1.0)))
                    qc.ry(theta, qr[i])

        qc.barrier()

        # --- Apply fuzzy rules as MCX gates ---
        for rule_str in self.rules:
            antecedents, consequent = self._parse_rule(rule_str, self.input_vars)

            # Collect control qubits
            controls = []
            for var_name, sname in antecedents:
                idx = self.input_fuzzysets[var_name]['set_names'].index(sname)
                controls.append(self._input_regs[var_name][idx])

            # Target qubit in the output register
            out_idx = self.output_fuzzysets[consequent[0]]['set_names'].index(
                consequent[1]
            )
            target = self._output_reg[out_idx]

            qc.mcx(controls, target)

        # --- Measurement on output qubits ---
        self._output_cr = ClassicalRegister(n_out, name='out')
        qc.add_register(self._output_cr)
        qc.barrier()
        for i in range(n_out):
            qc.measure(self._output_reg[i], self._output_cr[i])

        self._qc = qc

        if draw_qc:
            print(qc.draw(output='text'))

    # ------------------------------------------------------------------ #
    # Execution & defuzzification
    # ------------------------------------------------------------------ #

    def execute(self, n_shots=1024):
        """Run the circuit on a statevector-based sampler and defuzzify.

        Returns
        -------
        crisp_output : float
            Centroid-defuzzified crisp value.
        output_strengths : dict
            {set_name: activation_strength} measured from the circuit.
        """
        from qiskit.primitives import StatevectorSampler

        if self._sampler is None:
            self._sampler = StatevectorSampler()

        job = self._sampler.run([self._qc], shots=n_shots)
        result = job.result()

        # Retrieve counts for the 'out' classical register
        counts = result[0].data.out.get_counts()

        out_var = list(self.output_vars.keys())[0]
        set_names = self.output_fuzzysets[out_var]['set_names']
        n_out = len(set_names)

        # Marginal probability of each output qubit being |1⟩
        total_shots = sum(counts.values())
        output_strengths = {}
        for i, sname in enumerate(set_names):
            ones = 0
            for bitstring, count in counts.items():
                # Qiskit bit-order: rightmost char = qubit 0
                if len(bitstring) > i and bitstring[-(i + 1)] == '1':
                    ones += count
            output_strengths[sname] = ones / total_shots

        if self.verbose:
            print(f"Raw counts : {counts}")
            print(f"Output strengths: {output_strengths}")

        # --- Centroid defuzzification (Mamdani-style) ---
        universe = self.output_vars[out_var]
        mfs = self.output_fuzzysets[out_var]['mfs']

        # Clip each MF by its activation strength, then union (max)
        aggregated = np.zeros_like(universe, dtype=float)
        for i, sname in enumerate(set_names):
            strength = output_strengths[sname]
            clipped = np.minimum(mfs[i], strength)
            aggregated = np.maximum(aggregated, clipped)

        # Centroid
        total = np.sum(aggregated)
        if total < 1e-10:
            crisp_output = float(np.mean(universe))
        else:
            crisp_output = float(np.sum(universe * aggregated) / total)

        return crisp_output, output_strengths
