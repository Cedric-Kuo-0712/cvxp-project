import sys
import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from parser import parse_input
from partitioner import spectral_partition

def solve_placement_cvx(data, assignment, target_die, model_type="qp"):
    """
    Formulate and solve placement wirelength minimization on a single die using CVXPY.
    model_type: "lp" (Manhattan HPWL), "qp" (Quadratic L2-squared), or "socp" (Euclidean L2)
    """
    inst_names = [name for name, die in assignment.items() if die == target_die]
    inst_to_idx = {name: i for i, name in enumerate(inst_names)}
    n = len(inst_names)
    
    if n == 0:
        return {}
        
    tech_name = data.top_die_tech if target_die == 'top' else data.bottom_die_tech
    tech = data.technologies[tech_name]
    
    # Get cell areas and dimensions
    areas = []
    sizes = {}
    for name in inst_names:
        inst = data.instances[name]
        lc = tech.lib_cells[inst.lib_cell_name]
        areas.append(lc.size_x * lc.size_y)
        sizes[name] = (lc.size_x, lc.size_y)
    areas = np.array(areas, dtype=float)
    sum_areas = np.sum(areas)
    
    llx, lly, urx, ury = data.die_size
    center_x = (llx + urx) / 2.0
    center_y = (lly + ury) / 2.0
    
    # Variables
    x = cp.Variable(n)
    y = cp.Variable(n)
    
    # Constraints
    constraints = [
        # Center of gravity constraint
        areas @ x == center_x * sum_areas,
        areas @ y == center_y * sum_areas,
    ]
    
    # Boundary box constraints for each cell
    for i, name in enumerate(inst_names):
        w, h = sizes[name]
        constraints.append(x[i] >= llx)
        constraints.append(x[i] <= urx - w)
        constraints.append(y[i] >= lly)
        constraints.append(y[i] <= ury - h)
        
    # Virtual anchors at the four corners of the die to prevent collapsing
    anchors = {
        'C1': (llx, lly),
        'C8': (urx - tech.lib_cells[data.instances['C8'].lib_cell_name].size_x, lly),
        'C3': (urx - tech.lib_cells[data.instances['C3'].lib_cell_name].size_x, ury - tech.lib_cells[data.instances['C3'].lib_cell_name].size_y),
        'C6': (llx, ury - tech.lib_cells[data.instances['C6'].lib_cell_name].size_y)
    }
    
    # Objective
    obj_terms = []
    
    # Add anchor terms to objectives
    for name, (ax_val, ay_val) in anchors.items():
        if name in inst_to_idx:
            idx = inst_to_idx[name]
            # We weight the anchor more to force spreading in LP and SOCP
            weight = 5.0
            if model_type == "lp":
                obj_terms.append(weight * (cp.abs(x[idx] - ax_val) + cp.abs(y[idx] - ay_val)))
            elif model_type == "qp":
                obj_terms.append(weight * (cp.square(x[idx] - ax_val) + cp.square(y[idx] - ay_val)))
            elif model_type == "socp":
                obj_terms.append(weight * cp.norm(cp.vstack([x[idx] - ax_val, y[idx] - ay_val]), 2))
    
    if model_type == "lp":
        # Formulate L1 HPWL using helper variables
        # For each net, we want to minimize: (max_x - min_x) + (max_y - min_y)
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
            if len(die_pins) <= 1:
                continue
                
            idxs = [inst_to_idx[p[0]] for p in die_pins]
            
            # Helper variables for bounding box
            ux = cp.Variable()
            vx = cp.Variable()
            uy = cp.Variable()
            vy = cp.Variable()
            
            for idx in idxs:
                constraints.append(ux >= x[idx])
                constraints.append(vx <= x[idx])
                constraints.append(uy >= y[idx])
                constraints.append(vy <= y[idx])
                
            obj_terms.append((ux - vx) + (uy - vy))
            
        objective = cp.Minimize(cp.sum(obj_terms))
        
    elif model_type == "qp":
        # Minimize sum of squared Euclidean distances (Quadratic QP)
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
            d = len(die_pins)
            if d <= 1:
                continue
                
            idxs = [inst_to_idx[p[0]] for p in die_pins]
            weight = 1.0 / (d - 1)
            
            # Cliques expansion
            for i in range(d):
                for j in range(i + 1, d):
                    u_idx = idxs[i]
                    v_idx = idxs[j]
                    obj_terms.append(weight * (cp.square(x[u_idx] - x[v_idx]) + cp.square(y[u_idx] - y[v_idx])))
                    
        objective = cp.Minimize(cp.sum(obj_terms))
        
    elif model_type == "socp":
        # Minimize sum of Euclidean distances (Second-Order Cone Program)
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
            d = len(die_pins)
            if d <= 1:
                continue
                
            idxs = [inst_to_idx[p[0]] for p in die_pins]
            weight = 1.0 / (d - 1)
            
            for i in range(d):
                for j in range(i + 1, d):
                    u_idx = idxs[i]
                    v_idx = idxs[j]
                    # L2-norm of coordinate difference vector (x_diff, y_diff)
                    diff = cp.vstack([x[u_idx] - x[v_idx], y[u_idx] - y[v_idx]])
                    obj_terms.append(weight * cp.norm(diff, 2))
                    
        objective = cp.Minimize(cp.sum(obj_terms))
        
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    
    if prob.status not in ["optimal", "feasible"]:
        # Fallback solver
        prob.solve(solver=cp.SCS)
        
    # Return positions dict
    return {inst_names[i]: (x.value[i], y.value[i]) for i in range(n)}

def calculate_hpwl(data, assignment, target_die, positions):
    hpwl = 0.0
    for net_name, net in data.nets.items():
        die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
        if len(die_pins) <= 1:
            continue
        coords_x = [positions[p[0]][0] for p in die_pins]
        coords_y = [positions[p[0]][1] for p in die_pins]
        hpwl += (max(coords_x) - min(coords_x)) + (max(coords_y) - min(coords_y))
    return hpwl

def calculate_euclidean_wirelength(data, assignment, target_die, positions):
    eucl = 0.0
    for net_name, net in data.nets.items():
        die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
        d = len(die_pins)
        if d <= 1:
            continue
        weight = 1.0 / (d - 1)
        for i in range(d):
            for j in range(i + 1, d):
                pi = die_pins[i][0]
                pj = die_pins[j][0]
                dx = positions[pi][0] - positions[pj][0]
                dy = positions[pi][1] - positions[pj][1]
                eucl += weight * np.sqrt(dx**2 + dy**2)
    return eucl

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 solve_socp.py <input_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    data = parse_input(input_file)
    # Fix the partition for comparison
    assignment = spectral_partition(data)
    
    print("\n=== Solving Convex Placement Formulations via CVXPY ===")
    
    results = {}
    models = ["lp", "qp", "socp"]
    
    for m in models:
        print(f"  Solving {m.upper()} formulation on Top Die...")
        pos = solve_placement_cvx(data, assignment, 'top', model_type=m)
        print(f"    Positions: {pos}")
        hpwl = calculate_hpwl(data, assignment, 'top', pos)
        eucl = calculate_euclidean_wirelength(data, assignment, 'top', pos)
        results[m] = {
            "positions": pos,
            "hpwl": hpwl,
            "eucl": eucl
        }
        
    print("\n### 凸優化模型對比表格 (Convex Formulations Comparison)")
    print()
    print("| 指標 (Metrics) | LP (L1 HPWL) | QP (L2-Squared) | SOCP (L2 Euclidean) |")
    print("|---|---|---|---|")
    print(f"| **HPWL (L1 Wirelength)** | {results['lp']['hpwl']:.2f} | {results['qp']['hpwl']:.2f} | {results['socp']['hpwl']:.2f} |")
    print(f"| **Euclidean Wirelength** | {results['lp']['eucl']:.2f} | {results['qp']['eucl']:.2f} | {results['socp']['eucl']:.2f} |")
    print()
    
    # Plot side-by-side comparison using matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    llx, lly, urx, ury = data.die_size
    
    for i, m in enumerate(models):
        ax = axes[i]
        pos = results[m]["positions"]
        
        # Plot boundaries
        ax.plot([llx, urx, urx, llx, llx], [lly, lly, ury, ury, lly], color='black', linestyle='--', linewidth=1.5)
        
        # Plot instances as points/rectangles
        xs = [pos[name][0] for name in pos]
        ys = [pos[name][1] for name in pos]
        ax.scatter(xs, ys, color='#2ca02c', s=120, zorder=3, label='Cells')
        
        # Annotate cell names
        for name, (cx, cy) in pos.items():
            ax.annotate(name, (cx, cy), textcoords="offset points", xytext=(0,6), ha='center', fontsize=9, fontweight='bold')
            
        # Draw nets
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == 'top']
            if len(die_pins) <= 1:
                continue
            for j in range(1, len(die_pins)):
                p1 = die_pins[0][0]
                p2 = die_pins[j][0]
                ax.plot([pos[p1][0], pos[p2][0]], [pos[p1][1], pos[p2][1]], color='#1f77b4', alpha=0.6, zorder=2)
                
        ax.set_title(f"{m.upper()} Model\nHPWL: {results[m]['hpwl']:.2f} | L2: {results[m]['eucl']:.2f}", fontsize=11, fontweight='bold')
        ax.set_xlim(llx - 2, urx + 2)
        ax.set_ylim(lly - 2, ury + 2)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        
    plt.suptitle("3D Placement: LP vs. QP vs. SOCP Exact Solutions Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    image_name = "convex_formulations_comparison.png"
    plt.savefig(image_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot successfully saved as {image_name}")

if __name__ == "__main__":
    main()
