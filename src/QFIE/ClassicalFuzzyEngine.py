"""
Classical Fuzzy Inference Engine (CFIE)
=======================================
A pure-NumPy Mamdani-style fuzzy inference engine for comparison
against the Quantum Fuzzy Inference Engine (QFIE).

Uses the exact same:
  - Membership functions (trimf / trapmf)
  - Rule format
  - Centroid defuzzification

Only the rule evaluation differs:
  Classical  →  min() for AND, max() for OR, clip for implication
  Quantum    →  MCX gates + measurement statistics
"""

import numpy as np
from .FuzzyEngines import trimf, trapmf   # reuse the same MF helpers


class ClassicalFuzzyEngine:
    """Classical Mamdani fuzzy inference engine (no quantum)."""

    def __init__(self, verbose=False):
        self.verbose = verbose

        self.input_vars = {}
        self.output_vars = {}
        self.input_fuzzysets = {}
        self.output_fuzzysets = {}
        self.rules = []

    # ── Variable / set registration (same API as QFIE) ───────────────

    def input_variable(self, name, universe):
        self.input_vars[name] = np.asarray(universe, dtype=float)

    def output_variable(self, name, universe):
        self.output_vars[name] = np.asarray(universe, dtype=float)

    def add_input_fuzzysets(self, var_name, set_names, membership_fns):
        self.input_fuzzysets[var_name] = {
            'set_names': list(set_names),
            'mfs': list(membership_fns),
        }

    def add_output_fuzzysets(self, var_name, set_names, membership_fns):
        self.output_fuzzysets[var_name] = {
            'set_names': list(set_names),
            'mfs': list(membership_fns),
        }

    def set_rules(self, rules):
        self.rules = list(rules)

    # ── Fuzzification ────────────────────────────────────────────────

    def _fuzzify(self, crisp_inputs):
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

    # ── Rule parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_rule(rule_str, input_vars):
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
                consequent = (tokens[i + 1], tokens[i + 3])
                break
            if tok in input_vars:
                antecedents.append((tok, tokens[i + 2]))
                i += 3
                continue
            i += 1
        return antecedents, consequent

    # ── Classical Mamdani inference ──────────────────────────────────

    def infer(self, crisp_inputs):
        """Run classical fuzzy inference.

        Returns
        -------
        crisp_output : float
        output_strengths : dict  {set_name: activation_strength}
        """
        membership = self._fuzzify(crisp_inputs)

        if self.verbose:
            print("Membership degrees:")
            for var, degs in membership.items():
                print(f"  {var}: {degs}")

        out_var = list(self.output_vars.keys())[0]
        set_names = self.output_fuzzysets[out_var]['set_names']
        mfs = self.output_fuzzysets[out_var]['mfs']
        universe = self.output_vars[out_var]

        # Accumulate rule strengths per output set
        strengths = {sname: 0.0 for sname in set_names}

        for rule_str in self.rules:
            antecedents, consequent = self._parse_rule(rule_str, self.input_vars)

            # AND = min of antecedent membership degrees
            firing_strength = 1.0
            for var_name, sname in antecedents:
                firing_strength = min(firing_strength, membership[var_name][sname])

            # OR aggregation = max across rules targeting the same output set
            out_set = consequent[1]
            strengths[out_set] = max(strengths[out_set], firing_strength)

        if self.verbose:
            print(f"Output strengths: {strengths}")

        # ── Centroid defuzzification (Mamdani clip-then-aggregate) ───
        aggregated = np.zeros_like(universe, dtype=float)
        for i, sname in enumerate(set_names):
            clipped = np.minimum(mfs[i], strengths[sname])
            aggregated = np.maximum(aggregated, clipped)

        total = np.sum(aggregated)
        if total < 1e-10:
            crisp_output = float(np.mean(universe))
        else:
            crisp_output = float(np.sum(universe * aggregated) / total)

        return crisp_output, strengths
