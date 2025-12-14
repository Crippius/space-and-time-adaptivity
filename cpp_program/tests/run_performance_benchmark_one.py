
import subprocess
import re
import os
import csv
from prm_utils import ParameterFile

BASE_PRM = "../parameters_base.prm"
EXECUTABLE = "../build/main"
CSV_FILE = "performance_results_one.csv"

def parse_output(output):
    error = None
    time = None
    error_match = re.search(r"Final L2 Error\s*:\s*([0-9\.eE\-\+]+)", output)
    if error_match: error = float(error_match.group(1))
    time_match = re.search(r"Total Wall-clock time \(t\):\s*([0-9\.eE\-\+]+)", output)
    if time_match: time = float(time_match.group(1))
    return error, time


def run_benchmark(n_proc):
    results = []
    
    # Common helper
    def run_case(updates, label, category):
        prm = ParameterFile(BASE_PRM)
        for s, k, v in updates: prm.set_value(s, k, v)
        prm.write("temp_bench.prm")
        
        cmd = ["mpirun", "-np", str(n_proc), EXECUTABLE, "temp_bench.prm"]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            err, t = parse_output(res.stdout)
            if err and t:
                results.append([category, label, t, err])
                print(f"  -> Time: {t:.4f}, Error: {err:.4e}")
            else:
                print(f"  -> Failed to parse output")
        except Exception as e:
            print(f"  -> Failed: {e}")

    # # 1. Non-Adaptive
    # refinements = [2, 3, 4, 5]
    # dt = [0.01, 0.005, 0.001]
    # for r in refinements:
    #     for d in dt:
    #         updates = [
    #             ("Adaptivity Control", "Enable space adaptivity", "false"),
    #             ("Adaptivity Control", "Enable time adaptivity", "false"),
    #             ("Discretization", "Global refinements", str(r)),
    #             ("Discretization", "Initial deltat", str(d))
    #         ]
    #         label = f"Ref={r}, dt={d}"
    #         print(f"Running Non-Adaptive {label}...")
    #         run_case(updates, label, "Non-Adaptive")

    # 2. Adaptive (Space + Time)
    tolerances = [
        ("Loose", "0.005", "10", "0.001"), 
        ("Medium", "0.01", "5", "0.005"), 
        ("Tight", "0.002", "5", "0.010")
    ]
    
    for name, time_tol, ref_int, ref_frac in tolerances:
        updates = [
            ("Adaptivity Control", "Enable space adaptivity", "true"),
            ("Adaptivity Control", "Enable time adaptivity", "true"),
            ("Discretization", "Global refinements", "2"),
            ("Time Adaptivity", "Error upper bound", time_tol),
            ("Time Adaptivity", "Error lower bound", str(float(time_tol)/5.0)),
            ("Space Adaptivity", "Refinement interval", ref_int),
            ("Space Adaptivity", "Refinement fraction", ref_frac)
        ]
        label = f"S+T ({name})"
        print(f"Running Adaptive S+T {label}...")
        run_case(updates, label, "Adaptive (S+T)")

    # 3. Adaptive (Space Only)
    # Fixed small dt to minimize time error, but allow space adaptivity to work
    # space_confs = [
        # ("Loose", "10", "0.001"),
        # ("Medium", "5", "0.005"),
        # ("Tight", "5", "0.025")
    # ]
    
    # for name, ref_int, ref_frac in space_confs:
    #     updates = [
    #         ("Adaptivity Control", "Enable space adaptivity", "true"),
    #         ("Adaptivity Control", "Enable time adaptivity", "false"), # Disable time adapt
    #         ("Discretization", "Global refinements", "2"),
    #         ("Discretization", "Initial deltat", "0.01"), # Fixed dt
    #         ("Space Adaptivity", "Refinement interval", ref_int),
    #         ("Space Adaptivity", "Refinement fraction", ref_frac)
    #     ]
    #     label = f"Space ({name})"
    #     print(f"Running Adaptive Space-Only {label}...")
    #     run_case(updates, label, "Adaptive (Space)")

    # Write CSV
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Label", "Time", "Error"])
        writer.writerows(results)
    
    print(f"\nResults saved to {CSV_FILE}")
    
    # Try plotting
    try:
        import matplotlib.pyplot as plt
        
        non_adapt = [r for r in results if r[0] == "Non-Adaptive"]
        adapt_st = [r for r in results if r[0] == "Adaptive (S+T)"]
        adapt_s  = [r for r in results if r[0] == "Adaptive (Space)"]
        
        plt.figure(figsize=(10, 6))
        if non_adapt:
            plt.plot([r[2] for r in non_adapt], [r[3] for r in non_adapt], 'o-', label='Non-Adaptive')
        if adapt_st:
            plt.plot([r[2] for r in adapt_st], [r[3] for r in adapt_st], 's-', label='Adaptive (Space+Time)')
        if adapt_s:
            plt.plot([r[2] for r in adapt_s], [r[3] for r in adapt_s], '^-', label='Adaptive (Space Only)')
        
        plt.xlabel('Time (s)')
        plt.ylabel('L2 Error')
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Error vs Time: Adaptivity Strategies')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        plt.savefig('performance_plot_one.png')
        print("Plot saved to performance_plot_one.png")
    except ImportError:
        print("matplotlib not found, skipping plot generation. Use the CSV data to plot.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run performance benchmark')
    parser.add_argument('--np', type=int, default=4, help='Number of MPI processes')
    args = parser.parse_args()
    
    print(f"=== Performance Benchmark (MPI np={args.np}) ===")
    run_benchmark(args.np)

