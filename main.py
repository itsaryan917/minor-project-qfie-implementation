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
    print("  drone              Run drone altitude stabilization (QFIE)")
    print("  drone-compare      Compare classical vs quantum drone controller")
    print("  benchmark          Run large-scale benchmark")
    print("  cancer-sim         Run cancer drug delivery quantum fuzzy PI simulation")
    print("  cancer-compare     Compare classical vs quantum fuzzy PI (cancer)")
    print("  cancer-interactive Interactive controller selection and analysis")
    print()


def cancer_interactive():
    """Interactive menu for cancer controller selection and analysis."""
    print("\n" + "="*70)
    print("  CANCER DRUG DELIVERY — INTERACTIVE CONTROLLER ANALYSIS")
    print("="*70)
    
    while True:
        print("\nSelect a mode:")
        print("  1. Quantum Fuzzy PI Controller (QFIE)")
        print("  2. Hybrid PI + Quantum Fuzzy Controller")
        print("  3. Comparison of both controllers")
        print("  4. Run both sequentially")
        print("  5. Exit")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\nRunning Quantum Fuzzy PI Controller simulation...")
            from cancer_pi_controller_simulation import main as cancer_sim_main
            cancer_sim_main(mode='fuzzy-only')
            
        elif choice == "2":
            print("\nRunning Hybrid PI + Quantum Fuzzy Controller simulation...")
            from cancer_pi_controller_simulation import main as cancer_sim_main
            cancer_sim_main(mode='hybrid')
            
        elif choice == "3":
            print("\nRunning Comparison analysis...")
            from cancer_pi_controller_comparison import main as cancer_compare_main
            cancer_compare_main()
        
        elif choice == "4":
            print("\nRunning both controllers sequentially...\n")
            
            print("-"*70)
            print("PART 1: Quantum Fuzzy PI Controller")
            print("-"*70)
            from cancer_pi_controller_simulation import main as cancer_sim_main
            cancer_sim_main(mode='fuzzy-only')
            
            print("\n" + "-"*70)
            print("PART 2: Hybrid PI + Quantum Fuzzy Controller")
            print("-"*70)
            cancer_sim_main(mode='hybrid')
            
            print("\n" + "-"*70)
            print("PART 3: Comparison Analysis")
            print("-"*70)
            from cancer_pi_controller_comparison import main as cancer_compare_main
            cancer_compare_main()
        
        elif choice == "5":
            print("\nExiting interactive mode.\n")
            break
        
        else:
            print("\nInvalid choice. Please enter 1-5.")


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
    
    elif command in ("cancer-interactive", "cancer_interactive"):
        cancer_interactive()

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
