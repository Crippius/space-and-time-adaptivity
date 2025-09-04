import os
import subprocess
import re
import csv
from pathlib import Path
import sys
import time

# ====================================================================
# 1. CONFIGURAZIONE DEI PERCORSI E DEI TEST
# ====================================================================

# Percorso dell'eseguibile C++
EXECUTABLE_PATH = Path("./cpp_program/build/main").resolve()

# Percorso dello script per il calcolo dell'errore
ERROR_SCRIPT_PATH = Path("./compute_error.py").resolve()

# Percorso della soluzione di riferimento ("golden")
GOLDEN_REFERENCE_PATH = Path("./golden_reference/ref_7_0.0005/solution_9999.pvtu").resolve()

# Cartella principale dove verranno salvati tutti i risultati
RUNS_DIR = Path("./test_runs").resolve()

# File di parametri di base da cui partire
BASE_PARAMETERS_FILE = Path("./parameters_base.txt").resolve()

# Nome del file CSV di output
RESULTS_CSV_PATH = RUNS_DIR / "results.csv"


# --- DEFINIZIONE DEI TEST DA ESEGUIRE ---
test_configurations = [
    {
        "run_id": "high_refinement_adaptive",
        "Adaptivity Control": {"Enable space adaptivity": "true"},
        "Discretization": {"Global refinements": "4"},
        "Space Adaptivity": {"Refinement fraction": "0.2", "Coarsening fraction": "0.8"},
    },
    {
        "run_id": "low_refinement_adaptive",
        "Adaptivity Control": {"Enable space adaptivity": "true"},
        "Discretization": {"Global refinements": "2"},
        "Space Adaptivity": {"Refinement fraction": "0.1", "Coarsening fraction": "0.9"},
    },
    {
        "run_id": "uniform_mesh_no_adapt",
        "Adaptivity Control": {"Enable space adaptivity": "false"},
        "Discretization": {"Global refinements": "3"},
    },
    # Aggiungi qui altre configurazioni...
]


# ====================================================================
# 2. FUNZIONI HELPER
# ====================================================================

def modify_parameter_file(base_params_path, output_path, changes):
    """Crea un nuovo file di parametri con una logica di parsing corretta."""
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
    print(f"  -> File parametri creato in: {output_path}")

def parse_metrics_from_log(log_path):
    """Analizza il log della simulazione C++ e estrae le metriche."""
    metrics = {}
    patterns = {
        "total_time": r"Total Wall-clock time \(t\):\s+([\d\.\-eE]+)",
        "n_dofs": r"Final Degrees of Freedom \(n_Omega\):\s+([\d\.\-eE]+)",
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

def parse_all_parameters_from_prm(prm_path):
    """
    Legge un file .prm e estrae tutti i parametri definiti,
    creando un dizionario.
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
        print(f"  -> ERRORE: Impossibile leggere il file dei parametri {prm_path}")
        return {}
        
    return params

def run_error_computation(run_dir, reference_path):
    """
    Esegue lo script di calcolo dell'errore, che scrive il risultato su un file temporaneo.
    Questo script poi legge e cancella il file.
    """
    # Prefer parallel output .pvtu (MPI), fallback to single .vtu (serial)
    pvtu_path = run_dir / "output_9999.pvtu"
    vtu_path = run_dir / "output_9999.vtu"
    test_solution_path = pvtu_path if pvtu_path.exists() else vtu_path
    result_file_path = run_dir / "error_result.tmp"

    if not test_solution_path.exists():
        print(f"  -> ERRORE: File soluzione 'output_9999.vtu' non trovato in {run_dir}")
        return None
    
    command = [
        sys.executable,
        str(ERROR_SCRIPT_PATH),
        "--reference", str(reference_path),
        "--test-solution", str(test_solution_path),
        "--output-file", str(result_file_path)
    ]
    
    print("  -> Calcolo dell'errore L2 (tramite file temporaneo)...")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=300 # Aumentato a 5 minuti per sicurezza
        )
        if not result_file_path.exists():
            print("  -> ERRORE: Il file di risultato non è stato creato dallo script di errore.")
            return None
        with open(result_file_path, 'r') as f:
            l2_error = float(f.read().strip())
        print(f"  -> Errore L2 calcolato: {l2_error}")
        return l2_error
    except subprocess.TimeoutExpired:
        print("  -> ERRORE: Il calcolo dell'errore ha superato il tempo limite.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  -> ERRORE: Lo script di calcolo dell'errore è fallito (codice {e.returncode}).")
        print(f"--- Messaggio di errore dallo script:\n{e.stderr}\n---")
        return None
    except (ValueError, IndexError):
        print("  -> ERRORE: Il file di risultato non conteneva un numero valido.")
        return None
    except Exception as e:
        print(f"  -> ERRORE imprevisto durante il calcolo dell'errore: {e}")
        return None
    finally:
        if result_file_path.exists():
            os.remove(result_file_path)

def write_results_to_csv(filepath, data_list):
    """Scrive la lista di dizionari in un file CSV."""
    if not data_list:
        print("Nessun dato da scrivere nel file CSV.")
        return
    
    # Raccoglie tutti gli header possibili da tutti i dizionari per non perdere dati
    all_headers = set()
    for d in data_list:
        all_headers.update(d.keys())
    
    # Mantiene un ordine consistente
    ordered_headers = sorted(list(all_headers))

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_headers)
        writer.writeheader()
        writer.writerows(data_list)
    print(f"\n✅ Risultati finali salvati in: {filepath}")


# ====================================================================
# 3. LOGICA PRINCIPALE DELLA PIPELINE
# ====================================================================

def main():
    """Funzione principale che orchestra la pipeline di test."""
    RUNS_DIR.mkdir(exist_ok=True)
    all_results_data = []

    print("🚀 Inizio della pipeline di test...")

    for i, config in enumerate(test_configurations):
        run_name = f"run_{i:03d}_{config.get('run_id', 'test')}"
        print(f"\n--- Esecuzione Test: {run_name} ---")

        current_run_dir = RUNS_DIR / run_name
        current_run_dir.mkdir(exist_ok=True)
        
        params_path = current_run_dir / "parameters.prm"
        modify_parameter_file(BASE_PARAMETERS_FILE, params_path, config)
        
        print("  -> Avvio della simulazione C++...")
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
            print("  -> Simulazione completata con successo.")

            # Estrazione dei dati
            all_params = parse_all_parameters_from_prm(params_path)
            perf_metrics = parse_metrics_from_log(log_path)
            l2_error = run_error_computation(current_run_dir, GOLDEN_REFERENCE_PATH)
            
            # Calcolo delle metriche derivate
            derived_metrics = {}
            t = perf_metrics.get('total_time')
            n = perf_metrics.get('n_dofs')
            h = perf_metrics.get('h_min')
            e = l2_error

            derived_metrics['r_d_fixed'] = (e / n) if e is not None and n is not None and n > 0 else None
            derived_metrics['r_t_to_sol'] = (t / e) if t is not None and e is not None and e > 0 else None
            derived_metrics['r_eff_res'] = (h / e) if h is not None and e is not None and e > 0 else None
            
            # Unione di tutti i dati in un unico record
            current_run_results = {
                "run_name": run_name,
                **all_params,
                **perf_metrics,
                "l2_error": l2_error,
                **derived_metrics
            }
            all_results_data.append(current_run_results)

        except subprocess.CalledProcessError as e:
            print(f"  -> ERRORE: La simulazione C++ è fallita.")
            print(f"--- Stderr:\n{e.stderr}\n---------------")
            continue
        
    write_results_to_csv(RESULTS_CSV_PATH, all_results_data)

if __name__ == "__main__":
    if not all([p.exists() for p in [EXECUTABLE_PATH, ERROR_SCRIPT_PATH, GOLDEN_REFERENCE_PATH, BASE_PARAMETERS_FILE]]):
        print("ERRORE: Uno o più file/percorsi essenziali non sono stati trovati.")
        print(f"Controlla che esistano:\n- Eseguibile: {EXECUTABLE_PATH}\n- Script Errore: {ERROR_SCRIPT_PATH}\n- Riferimento: {GOLDEN_REFERENCE_PATH}\n- Parametri Base: {BASE_PARAMETERS_FILE}")
    else:
        main()
