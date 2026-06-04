import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

from parser import parse_input
from partitioner import spectral_partition
from nesterov_placer import run_nesterov_placer
from legalizer import legalize_cells_on_rows, optimize_and_legalize_terminals
from main import calculate_hpwl_3d

@contextmanager
def silence_stdout():
    new_target = open(os.devnull, 'w')
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target
        new_target.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_pareto.py <input_file> [--density-solver {fft,dct}]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    solver_type = "fft"
    if "--density-solver" in sys.argv:
        idx = sys.argv.index("--density-solver")
        if idx + 1 < len(sys.argv):
            solver_type = sys.argv[idx + 1]
            
    wl_model = "quadratic"
    if "--wirelength-model" in sys.argv:
        idx = sys.argv.index("--wirelength-model")
        if idx + 1 < len(sys.argv):
            wl_model = sys.argv[idx + 1]
            
    gamma_val = 0.5
    if "--gamma" in sys.argv:
        idx = sys.argv.index("--gamma")
        if idx + 1 < len(sys.argv):
            gamma_val = float(sys.argv[idx + 1])
            
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    print(f"=== Starting Pareto Frontier Sweep ===")
    print(f"Input file: {input_file}")
    print(f"Density Solver: {solver_type.upper()}")
    print(f"Wirelength Model: {wl_model.upper()} (gamma={gamma_val})")
    
    # 1. Parse Input
    data = parse_input(input_file)
    
    # Calculate cell areas to verify utilization
    top_tech = data.technologies[data.top_die_tech]
    bottom_tech = data.technologies[data.bottom_die_tech]
    
    inst_names = list(data.instances.keys())
    
    llx, lly, urx, ury = data.die_size
    die_area = (urx - llx) * (ury - lly)
    max_area_top = die_area * (data.top_die_max_util / 100.0)
    max_area_bottom = die_area * (data.bottom_die_max_util / 100.0)
    
    # Sweep ratios from 0.15 to 0.85 in steps of 0.05
    ratios = np.arange(0.15, 0.86, 0.05)
    
    results = []
    
    print("\nSweeping partition ratios...")
    print(f"{'Ratio':<10}{'Top Util %':<12}{'Bottom Util %':<15}{'HPWL':<10}{'Terminals':<12}{'Score':<10}{'Status':<10}")
    print("-" * 84)
    
    for ratio in ratios:
        try:
            with silence_stdout():
                # 2. Partition
                assignment = spectral_partition(data, target_top_ratio=ratio)
                
                # Check area utilization constraints
                area_t = sum(top_tech.lib_cells[data.instances[name].lib_cell_name].size_x * 
                             top_tech.lib_cells[data.instances[name].lib_cell_name].size_y 
                             for name, die in assignment.items() if die == 'top')
                             
                area_b = sum(bottom_tech.lib_cells[data.instances[name].lib_cell_name].size_x * 
                             bottom_tech.lib_cells[data.instances[name].lib_cell_name].size_y 
                             for name, die in assignment.items() if die == 'bottom')
                             
                util_t = (area_t / die_area) * 100.0
                util_b = (area_b / die_area) * 100.0
                
                is_valid = (area_t <= max_area_top) and (area_b <= max_area_bottom)
                
                # 3. Global Placement
                gp_positions = {}
                for die in ['top', 'bottom']:
                    positions_die = run_nesterov_placer(data, assignment, die, num_iterations=50, lr=0.5,
                                                        density_solver=solver_type,
                                                        wirelength_model=wl_model, gamma=gamma_val)
                    gp_positions.update(positions_die)
                    
                # 4. Legalize
                legal_positions = legalize_cells_on_rows(data, assignment, gp_positions)
                
                # 5. Terminals
                legal_terminals = optimize_and_legalize_terminals(data, assignment, legal_positions)
                
                # 6. Final Metrics
                hpwl = calculate_hpwl_3d(data, assignment, legal_positions, legal_terminals)
                num_terminals = len(legal_terminals)
                terminal_cost = num_terminals * data.terminal_cost
                total_score = hpwl + terminal_cost
                
            status_str = "Valid" if is_valid else "Violated"
            print(f"{ratio:<10.2f}{util_t:<12.2f}{util_b:<15.2f}{hpwl:<10.2f}{num_terminals:<12d}{total_score:<10.2f}{status_str:<10}")
            
            if is_valid:
                results.append((hpwl, num_terminals, ratio, total_score, util_t, util_b))
                
        except Exception as e:
            # Re-raise or print error
            print(f"{ratio:<10.2f}Error: {e}")
            
    if not results:
        print("\nNo valid partitioning results found within utilization limits.")
        sys.exit(1)
        
    # Find Pareto Frontier
    # Sort by number of terminals, then by hpwl
    sorted_res = sorted(results, key=lambda x: (x[1], x[0]))
    pareto_points = []
    for pt in sorted_res:
        if not pareto_points:
            pareto_points.append(pt)
        else:
            # If current hpwl is strictly smaller than the last added pareto point, it is not dominated
            if pt[0] < pareto_points[-1][0]:
                pareto_points.append(pt)
                
    print("\n=== Pareto-Optimal Solutions ===")
    print(f"{'Ratio':<10}{'Top Util %':<12}{'Bottom Util %':<15}{'HPWL':<10}{'Terminals':<12}{'Score':<10}")
    print("-" * 74)
    for hpwl, num_terminals, ratio, score, util_t, util_b in sorted(pareto_points, key=lambda x: x[1]):
        print(f"{ratio:<10.2f}{util_t:<12.2f}{util_b:<15.2f}{hpwl:<10.2f}{num_terminals:<12d}{score:<10.2f}")
        
    # Plot Pareto Frontier
    plt.figure(figsize=(9, 6.5))
    
    # Plot all valid runs
    all_hpwls = [r[0] for r in results]
    all_terms = [r[1] for r in results]
    plt.scatter(all_terms, all_hpwls, color='#1f77b4', marker='o', s=80, alpha=0.5, label='Valid Configurations')
    
    # Plot Pareto Frontier
    pareto_sorted = sorted(pareto_points, key=lambda x: x[1])
    p_hpwls = [pt[0] for pt in pareto_sorted]
    p_terms = [pt[1] for pt in pareto_sorted]
    
    plt.plot(p_terms, p_hpwls, color='#d62728', marker='s', markersize=8, linestyle='-', linewidth=2.5, label='Pareto Frontier')
    
    # Annotate Pareto points
    for hpwl, num_terminals, ratio, _, _, _ in pareto_sorted:
        plt.annotate(f"r={ratio:.2f}", (num_terminals, hpwl), textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold', color='#d62728')
        
    plt.xlabel('Number of Terminals (Cross-Die Nets)', fontsize=12)
    plt.ylabel('Total HPWL (Wirelength)', fontsize=12)
    plt.title(f'3D Placement Pareto Frontier: HPWL vs. Terminals ({solver_type.upper()}, {wl_model.upper()})', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    image_name = f"pareto_frontier_{solver_type}_{wl_model}.png"
    plt.savefig(image_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPareto Frontier visualization saved as {image_name}")

if __name__ == "__main__":
    main()
