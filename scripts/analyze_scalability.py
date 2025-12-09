#!/usr/bin/env python3
"""
Scalability Analysis Script
Analyzes strong scaling results and generates CSV with performance metrics.

Usage: python analyze_scalability.py <results_directory>
"""

import os
import sys
import csv
from pathlib import Path
import re


def extract_walltime(proc_dir):
    """Extract walltime from a processor run directory."""
    walltime_file = proc_dir / "walltime.txt"
    
    if not walltime_file.exists():
        return None
    
    try:
        with open(walltime_file, 'r') as f:
            content = f.read().strip()
            if content == "ERROR":
                return None
            return float(content)
    except (ValueError, IOError):
        return None


def extract_iterations(proc_dir):
    """Extract number of time steps from output log."""
    output_log = proc_dir / "output.log"
    
    if not output_log.exists():
        return None
    
    try:
        with open(output_log, 'r') as f:
            content = f.read()
        
        # Count lines that match the time step pattern: "n = XX, t = ..."
        # Look for accepted steps (not rejected ones)
        time_step_matches = re.findall(r'^n\s*=\s*(\d+),\s*t\s*=', content, re.MULTILINE)
        
        if time_step_matches:
            # Return the maximum step number (last step)
            return max(int(n) for n in time_step_matches)
        
        return None
    except IOError:
        return None


def extract_dofs(proc_dir):
    """Extract final number of degrees of freedom from output log."""
    output_log = proc_dir / "output.log"
    
    if not output_log.exists():
        return None
    
    try:
        with open(output_log, 'r') as f:
            content = f.read()
        
        # Try to find final DOF count from Performance Metrics Summary
        # Pattern: "Final Degrees of Freedom (n_Omega):     7.615e+04"
        match = re.search(r'Final Degrees of Freedom.*?:\s*([\d.]+)e\+(\d+)', content, re.IGNORECASE)
        if match:
            base = float(match.group(1))
            exponent = int(match.group(2))
            return int(base * (10 ** exponent))
        
        # Fallback: try to find initial DOF count
        match = re.search(r'Number of DoFs\s*=\s*(\d+)', content, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    except IOError:
        return None


def analyze_scalability(results_dir):
    """
    Analyze scalability results and generate CSV.
    
    Args:
        results_dir: Path to the results directory containing nprocs_* subdirectories
    
    Returns:
        Dictionary with analysis results
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return None
    
    # Find all processor count directories
    proc_dirs = sorted(results_path.glob("nprocs_*"))
    
    if not proc_dirs:
        print(f"Error: No nprocs_* directories found in {results_dir}")
        return None
    
    # Collect data
    results = []
    
    for proc_dir in proc_dirs:
        # Extract number of processors from directory name
        nprocs_str = proc_dir.name.replace("nprocs_", "")
        try:
            nprocs = int(nprocs_str)
        except ValueError:
            print(f"Warning: Could not parse processor count from {proc_dir.name}")
            continue
        
        # Extract walltime
        walltime = extract_walltime(proc_dir)
        
        if walltime is None:
            print(f"Warning: Could not extract walltime for {nprocs} processors")
            continue
        
        # Extract additional info
        iterations = extract_iterations(proc_dir)
        dofs = extract_dofs(proc_dir)
        
        results.append({
            'nprocs': nprocs,
            'walltime': walltime,
            'iterations': iterations,
            'dofs': dofs
        })
    
    if not results:
        print("Error: No valid results found")
        return None
    
    # Sort by number of processors
    results.sort(key=lambda x: x['nprocs'])
    
    # Calculate speedup and efficiency (relative to serial run)
    if results[0]['nprocs'] == 1:
        serial_time = results[0]['walltime']
    else:
        # If no serial run, use first available as reference
        serial_time = results[0]['walltime']
        print(f"Warning: No serial (nprocs=1) run found. Using {results[0]['nprocs']} procs as reference.")
    
    for result in results:
        result['speedup'] = serial_time / result['walltime']
        result['efficiency'] = result['speedup'] / result['nprocs'] * 100  # in percentage
        result['time_per_iteration'] = result['walltime'] / result['iterations'] if result['iterations'] else None
    
    # Save to CSV
    csv_path = results_path / "scalability_results.csv"
    
    fieldnames = ['nprocs', 'walltime', 'speedup', 'efficiency', 'time_steps', 
                  'time_per_step', 'final_dofs']
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Format the output nicely
            row = {
                'nprocs': result['nprocs'],
                'walltime': f"{result['walltime']:.4f}",
                'speedup': f"{result['speedup']:.2f}",
                'efficiency': f"{result['efficiency']:.2f}",
                'time_steps': result['iterations'] if result['iterations'] is not None else 'N/A',
                'time_per_step': f"{result['time_per_iteration']:.6f}" if result['time_per_iteration'] is not None else 'N/A',
                'final_dofs': result['dofs'] if result['dofs'] is not None else 'N/A',
            }
            writer.writerow(row)
    
    print(f"Results saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*90)
    print("SCALABILITY ANALYSIS SUMMARY")
    print("="*90)
    print(f"{'Procs':<8} {'Time(s)':<12} {'Speedup':<10} {'Efficiency(%)':<15} {'Steps':<10} {'Final DOFs':<12}")
    print("-"*90)
    
    for result in results:
        steps_str = str(result['iterations']) if result['iterations'] is not None else 'N/A'
        dofs_str = str(result['dofs']) if result['dofs'] is not None else 'N/A'
        print(f"{result['nprocs']:<8} {result['walltime']:<12.4f} "
              f"{result['speedup']:<10.2f} {result['efficiency']:<15.2f} {steps_str:<10} {dofs_str:<12}")
    
    print("="*90)
    
    # Print additional statistics
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    print(f"\nBest efficiency: {best_efficiency['efficiency']:.2f}% with {best_efficiency['nprocs']} processes")
    
    if len(results) > 1:
        max_speedup = results[-1]
        print(f"Maximum speedup: {max_speedup['speedup']:.2f}x with {max_speedup['nprocs']} processes")
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_scalability.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    results = analyze_scalability(results_dir)
    
    if results is None:
        sys.exit(1)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
