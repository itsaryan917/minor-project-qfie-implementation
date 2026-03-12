"""
Main entry point for all project scripts.
==========================================
Run with: uv run python main.py <command>

Available commands:
  drone           — Drone altitude stabilization (Quantum Fuzzy)
  drone-compare   — Drone: Classical vs Quantum comparison
  benchmark       — Large-scale benchmark
  cancer-sim      — Cancer drug delivery: Quantum Fuzzy PI simulation
  cancer-compare  — Cancer drug delivery: Classical vs Quantum comparison
"""

import sys


def print_usage():
    print(__doc__)
    print("Usage: python main.py <command>\n")
    print("Commands:")
    print("  drone            Run drone altitude stabilization (QFIE)")
    print("  drone-compare    Compare classical vs quantum drone controller")
    print("  benchmark        Run large-scale benchmark")
    print("  cancer-sim       Run cancer drug delivery quantum fuzzy PI simulation")
    print("  cancer-compare   Compare classical vs quantum fuzzy PI (cancer)")
    print()


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower().strip()

    if command == "drone":
        from stablizer import DroneAltitudeController
        controller = DroneAltitudeController()
        controller.run_simulation(error=-15, velocity=0, steps=150)

    elif command == "drone-compare":
        from comparison import main as comparison_main
        comparison_main()

    elif command == "benchmark":
        from large_scale_benchmark import main as benchmark_main
        benchmark_main()

    elif command in ("cancer-sim", "cancer_sim"):
        from cancer_pi_controller_simulation import main as cancer_sim_main
        cancer_sim_main()

    elif command in ("cancer-compare", "cancer_compare"):
        from cancer_pi_controller_comparison import main as cancer_compare_main
        cancer_compare_main()

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
