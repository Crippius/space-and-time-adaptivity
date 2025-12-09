import os
import subprocess
import re
import csv
from pathlib import Path
import sys
import time


### DEFINE PATHS

# Path to the C++ executable
EXECUTABLE_PATH = Path("../cpp_program/build/main").resolve()

# Path to the error computation script
ERROR_SCRIPT_PATH = Path("./compute_error.py").resolve()

# Percorso della soluzione di riferimento ("golden")
GOLDEN_REFERENCE_PATH = Path("../golden_reference/ref_7_0.0005/solution_9999.pvtu").resolve()

# Main directory where all results will be saved
RUNS_DIR = Path("../test_runs").resolve()

# Base parameter file to start from
BASE_PARAMETERS_FILE = Path("../parameters_base.prm").resolve()

# Output CSV file name
RESULTS_CSV_PATH = RUNS_DIR / "results.csv"


### SIMULATIONS TO TEST

test_configurations = []

for refinement in [1, 2, 3, 4, 5]:
    for space_adaptivity in [True, False]:
        for time_adaptivity in [True, False]:
            run_id = f"refinement_{refinement}_space_{'adaptive' if space_adaptivity else 'not_adaptive'}_time_{'adaptive' if time_adaptivity else 'not_adaptive'}"
            test_configurations.append({
                "run_id": run_id,
                "Adaptivity Control": {
                    "Enable space adaptivity": str(space_adaptivity).lower(),
                    "Enable time adaptivity": str(time_adaptivity).lower()
                },
                "Discretization": {"Global refinements": str(refinement)},
            })



### HELPER FUNCTIONS
# To work with parameter files

# Make changes to parameter files 
def modify_parameter_file(base_params_path, output_path, changes):
    """Creates a new parameter file."""
    with open(base_params_path, "r") as f_in:
        lines = f_in.readlines()

    modified_lines = []
    current_subsection = None

    for line in lines:
        stripped_line = line.strip()

        # Controlla i cambi di stato (inizio o fine di una sottosezione)
        if stripped_line.lower().startswith("subsection"):
            current_subsection = stripped_line.split(" ", 1)[-1].strip()
            modified_lines.append(line)
            continue
        
        if stripped_line.lower() == "end":
            current_subsection = None
            modified_lines.append(line)
            continue

        # Se siamo in una sottosezione di interesse, prova a modificare i parametri
        new_line = line
        if current_subsection and current_subsection in changes:
            for param, value in changes[current_subsection].items():
                # Cerca una riga che inizi con "set", seguita dal nome del parametro
                pattern = re.compile(f"^\\s*set\\s+{re.escape(param)}\\s*=", re.IGNORECASE)
                if pattern.search(stripped_line):
                    # Ricostruisce la riga per preservare l'indentazione e formattare correttamente
                    indentation = re.match(r"(\s*)", line).group(1)
                    new_line = f"{indentation}set {param} = {value}\n"
                    break  # Parametro trovato e sostituito, passa alla riga successiva
        
        modified_lines.append(new_line)

    with open(output_path, "w") as f_out:
        f_out.writelines(modified_lines)
    print(f"  -> Parameter file created at: {output_path}")

# Parses the C++ simulation log and extracts metrics
def parse_metrics_from_log(log_path):
    """Parses the C++ simulation log and extracts metrics."""
    metrics = {}
    patterns = {
        "total_time": r"Total Wall-clock time \(t\):\s+([\d\.\-eE]+)",
        "n_dofs": r"Maximum Degrees of Freedom \(n_Omega\):\s+([\d\.\-eE]+)",
        "h_min": r"Minimum cell diameter \(h\):\s+([\d\.\-eE]+)",
        "r_res": r"r_res \(h / n_Omega\):\s+([\d\.\-eE]+)",
        "r_t_per_dof": r"r_t-per-DOF \(t / n_Omega\):\s+([\d\.\-eE]+)",
    }
    try:
        with open(log_path, "r") as f:
            log_content = f.read()
        for key, pattern in patterns.items():
            match = re.search(pattern, log_content)
            metrics[key] = float(match.group(1)) if match else None
    except (IOError, ValueError):
        metrics = {key: None for key in patterns}
    return metrics

# Parses all parameters from a .prm file
def parse_all_parameters_from_prm(prm_path):
    """
    Reads a .prm file and extracts all defined parameters,
    creating a dictionary.
    """
    params = {}
    current_subsection = "General"
    param_pattern = re.compile(r"^\s*set\s+(.+?)\s*=\s*(.+?)\s*$")

    try:
        with open(prm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.lower().startswith("subsection"):
                    current_subsection = line.split(" ", 1)[-1].strip()
                    continue

                match = param_pattern.match(line)
                if match:
                    param_name = match.group(1).strip().replace(' ', '_')
                    param_value = match.group(2).strip()
                    # Crea una chiave univoca per evitare collisioni tra sezioni
                    full_key = f"param_{current_subsection.replace(' ', '_')}_{param_name}"
                    params[full_key] = param_value
    except IOError:
        print(f"  -> ERROR: Unable to read parameter file {prm_path}")
        return {}
        
    return params

# Runs the error computation script, which writes the result to a temporary file.
def run_error_computation(run_dir, reference_path):
    """
    Runs the error computation script, which writes the result to a temporary file.
    This script then reads and deletes the file.
    """
    # Prefer parallel output .pvtu (MPI), fallback to single .vtu (serial)
    pvtu_path = run_dir / "output_9999.pvtu"
    vtu_path = run_dir / "output_9999.vtu"
    test_solution_path = pvtu_path if pvtu_path.exists() else vtu_path
    result_file_path = run_dir / "error_result.tmp"

    if not test_solution_path.exists():
        print(f"  -> ERROR: Solution file 'output_9999.vtu' not found in {run_dir}")
        return None
    
    command = [
        sys.executable,
        str(ERROR_SCRIPT_PATH),
        "--reference", str(reference_path),
        "--test-solution", str(test_solution_path),
        "--output-file", str(result_file_path)
    ]
    
    print("  -> Computing L2 error (using temporary file)...")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300 # Aumentato a 5 minuti per sicurezza
        )
        if not result_file_path.exists():
            print("  -> ERROR: The result file was not created by the error script.")
            return None
        with open(result_file_path, 'r') as f:
            l2_error = float(f.read().strip())
        print(f"  -> Computed L2 error: {l2_error}")
        return l2_error
    except subprocess.TimeoutExpired:
        print("  -> ERROR: Error computation exceeded the time limit.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  -> ERROR: The error computation script failed (code {e.returncode}).")
        print(f"--- Error message from script:\n{e.stderr}\n---")
        return None
    except (ValueError, IndexError):
        print("  -> ERROR: The result file did not contain a valid number.")
        return None
    except Exception as e:
        print(f"  -> UNEXPECTED ERROR during error computation: {e}")
        return None
    finally:
        if result_file_path.exists():
            os.remove(result_file_path)

# Writes the results to a CSV file
def write_results_to_csv(filepath, data_list):
    """Writes the list of dictionaries to a CSV file."""
    if not data_list:
        print("No data to write to the CSV file.")
        return
    
    all_headers = set()
    for d in data_list:
        all_headers.update(d.keys())
    ordered_headers = sorted(list(all_headers))

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_headers)
        writer.writeheader()
        writer.writerows(data_list)
    print(f"\nFinal results saved in: {filepath}")



# MAIN
def main():
    """Main function that orchestrates the test pipeline."""
    RUNS_DIR.mkdir(exist_ok=True)
    all_results_data = []

    print("Starting the test pipeline...")

    for i, config in enumerate(test_configurations):
        run_name = f"run_{i:03d}_{config.get('run_id', 'test')}"
        print(f"\n--- Running Test: {run_name} ---")

        current_run_dir = RUNS_DIR / run_name
        current_run_dir.mkdir(exist_ok=True)
        
        params_path = current_run_dir / "parameters.prm"
        modify_parameter_file(BASE_PARAMETERS_FILE, params_path, config)
        
        print("  -> Starting C++ simulation...")
        command_str = f'"{str(EXECUTABLE_PATH)}" "{str(params_path)}"'
        log_path = current_run_dir / "stdout.log"

        try:
            result = subprocess.run(
                command_str,
                capture_output=True, text=True, check=True,
                cwd=str(current_run_dir), shell=True
            )
            with open(log_path, "w") as f_log:
                f_log.write(result.stdout)
            print("  -> Simulation completed successfully.")

            # Data extraction
            all_params = parse_all_parameters_from_prm(params_path)
            perf_metrics = parse_metrics_from_log(log_path)
            l2_error = run_error_computation(current_run_dir, GOLDEN_REFERENCE_PATH)
            
            # Compute derived metrics
            derived_metrics = {}
            t = perf_metrics.get('total_time')
            n = perf_metrics.get('n_dofs')
            h = perf_metrics.get('h_min')
            e = l2_error

            derived_metrics['r_d_fixed'] = (e / n) if e is not None and n is not None and n > 0 else None
            derived_metrics['r_t_to_sol'] = (t / e) if t is not None and e is not None and e > 0 else None
            derived_metrics['r_eff_res'] = (h / e) if h is not None and e is not None and e > 0 else None
            
            current_run_results = {
                "run_name": run_name,
                **all_params,
                **perf_metrics,
                "l2_error": l2_error,
                **derived_metrics
            }
            all_results_data.append(current_run_results)

        except subprocess.CalledProcessError as e:
            print(f"  -> ERROR: The C++ simulation failed.")
            print(f"--- Stderr:\n{e.stderr}\n---------------")
            continue
        
    write_results_to_csv(RESULTS_CSV_PATH, all_results_data)

if __name__ == "__main__":
    if not all([p.exists() for p in [EXECUTABLE_PATH, ERROR_SCRIPT_PATH, GOLDEN_REFERENCE_PATH, BASE_PARAMETERS_FILE]]):
        print("ERROR: One or more essential files/paths were not found.")
        print(f"Check that the following exist:\n- Executable: {EXECUTABLE_PATH}\n- Error Script: {ERROR_SCRIPT_PATH}\n- Reference: {GOLDEN_REFERENCE_PATH}\n- Base Parameters: {BASE_PARAMETERS_FILE}")
    else:
        main()
