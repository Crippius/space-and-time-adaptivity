#!/bin/bash

################################################################################
# Scalability Testing Script
# Tests strong scaling with different MPI process counts
################################################################################

set -e  # Exit on error

# Configuration
EXECUTABLE="../cpp_program/build/main"
PARAM_FILE="$(cd .. && pwd)/parameters_base.prm"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$(cd .. && pwd)/scalability_results/run_${TIMESTAMP}"
SCRIPTS_DIR="$(pwd)"

# MPI process counts to test (adjust based on your machine)
MPI_COUNTS=(1 2 4 8)
# MPI_COUNTS=(1 2 4 8 16 32 64 128)  # For 

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

################################################################################
# Setup
################################################################################

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Heat Solver - Scalability Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: Executable not found at $EXECUTABLE${NC}"
    echo "Please compile the program first:"
    echo "  cd cpp_program/build && cmake .. && make"
    exit 1
fi

# Check if parameter file exists
if [ ! -f "$PARAM_FILE" ]; then
    echo -e "${RED}Error: Parameter file not found at $PARAM_FILE${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$RUN_DIR"
echo -e "${GREEN}Created output directory: $RUN_DIR${NC}"
echo ""

# Copy parameter file to run directory
cp "$PARAM_FILE" "$RUN_DIR/parameters.prm"

################################################################################
# Run scalability tests
################################################################################

echo "Starting scalability tests..."
echo "MPI counts to test: ${MPI_COUNTS[*]}"
echo ""

for NPROCS in "${MPI_COUNTS[@]}"; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Testing with $NPROCS MPI processes${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Create subdirectory for this run
    PROC_DIR="$RUN_DIR/nprocs_${NPROCS}"
    mkdir -p "$PROC_DIR"
    
    # Change to the build directory (where the executable expects to run)
    cd ../cpp_program/build
    
    # Run the simulation and capture timing
    echo "Running simulation..."
    START_TIME=$(date +%s.%N)
    
    if mpirun -np "$NPROCS" ./main "$PARAM_FILE" > "$PROC_DIR/output.log" 2>&1; then
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
        
        echo -e "${GREEN}✓ Completed in ${ELAPSED} seconds${NC}"
        echo "$ELAPSED" > "$PROC_DIR/walltime.txt"
        
        # Move output files to the run directory
        if ls output_*.vtu output_*.pvtu 1> /dev/null 2>&1; then
            mv output_*.vtu output_*.pvtu "$PROC_DIR/" 2>/dev/null || true
        fi
    else
        echo -e "${RED}✗ Failed - check $PROC_DIR/output.log${NC}"
        echo "ERROR" > "$PROC_DIR/walltime.txt"
    fi
    
    # Return to scripts directory
    cd "$SCRIPTS_DIR"
    
    echo ""
done

################################################################################
# Analyze results
################################################################################

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Analyzing Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Run Python analysis script
if [ -f "./analyze_scalability.py" ]; then
    echo "Running analysis script..."
    python ./analyze_scalability.py "$RUN_DIR"
    echo ""
    
    if [ -f "$RUN_DIR/scalability_results.csv" ]; then
        echo -e "${GREEN}✓ Results saved to: $RUN_DIR/scalability_results.csv${NC}"
        echo ""
        echo "Summary:"
        column -t -s',' "$RUN_DIR/scalability_results.csv" 2>/dev/null || cat "$RUN_DIR/scalability_results.csv"
    fi
else
    echo -e "${RED}Warning: analyze_scalability.py not found${NC}"
    echo "Manual analysis needed."
fi

echo ""
echo -e "${GREEN}All tests completed!${NC}"
echo -e "Results directory: ${BLUE}$RUN_DIR${NC}"
