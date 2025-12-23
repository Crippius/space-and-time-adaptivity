#!/bin/bash
#SBATCH --job-name=pde_scaling_1to112
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --account=nct_345
#SBATCH --qos=gp_debug
#SBATCH --partition=gp

################################################################################
# SCALABILITY TEST CONFIGURATION SUMMARY
# ------------------------------------------------------------------------------
# Problem Setup:
#   - Spatial Peaks: 128
#   - Temporal Peaks: 5
#   - Initial Global Refinement: 6
# 
# Hardware/Partition:
#   - Cluster: MareNostrum (BSC)
#   - Partition: General Purpose (GP)
#   - Queue: gp_debug (Single Node)
#   - Scaling Range: 4 to 112 CPU Cores
################################################################################

# Purge existing modules and load Singularity for container execution
module purge
module load singularity

# --- Path Definitions ---
# SIF: Path to the Singularity Image File containing the compiled toolchain
# PRM: Path to the parameter file defining the PDE solver settings
# OUTDIR: Directory for logs, named with the unique Slurm Job ID
SIF="test6.sif"
PRM="../cpp_program/parameters_base.prm"
OUTDIR="scaling_internal_${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"

echo "=== STARTING STRONG SCALING TEST ==="
echo "Execution Node: $(hostname)"
echo "Core Configurations: 4, 8, 16, 32, 64, 112"
echo "---------------------------------------------------"

# Scaling Loop: Iterate through different numbers of MPI processes (P)
for P in 4 8 16 32 64 112
do
    echo ""
    echo ">>> RUNNING SIMULATION WITH P=$P CORES <<<"

    # Define log and error file paths for the current iteration
    RUN_LOG="${OUTDIR}/run_P${P}.out"
    RUN_ERR="${OUTDIR}/run_P${P}.err"

    # Execution via Singularity:
    # --cleanenv: Ensures a clean environment inside the container
    # mpirun -n $P: Launches P MPI processes
    # --bind-to core: Enforces process pinning to physical cores for optimal performance
    # /app/build/main: The solver binary inside the container image
    singularity exec --cleanenv "$SIF" \
        mpirun -n $P \
        --bind-to core \
        /app/build/main "$PRM" \
        > "$RUN_LOG" 2> "$RUN_ERR"

    # Quick check for completion
    echo "P=$P completed. Results stored in $RUN_LOG"
done

echo ""
echo "=== ALL TESTS COMPLETED ==="
echo "Summary of output files in: $OUTDIR"
ls -lh "$OUTDIR"