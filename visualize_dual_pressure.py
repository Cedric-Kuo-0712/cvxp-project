import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def solve_legalization_qp_with_dmax(x_star, widths, llx, urx, d_max=None):
    n = len(widths)
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(x - x_star))
    
    constraints = []
    constraints.append(x[0] >= llx)
    for i in range(n - 1):
        constraints.append(x[i+1] - x[i] >= widths[i])
    constraints.append(x[-1] + widths[-1] <= urx)
    
    if d_max is not None:
        for i in range(n):
            constraints.append(x[i] - x_star[i] <= d_max)
            constraints.append(x_star[i] - x[i] <= d_max)
            
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    return prob.status, prob.value, x.value

def main():
    print("=== Row Legalization QP & Dual Pressure Analysis ===")
    
    # 1. Problem Setup
    llx = 0.0
    urx = 50.0
    widths = np.array([4.0, 6.0, 5.0, 8.0, 4.0, 6.0, 5.0, 4.0])
    n = len(widths)
    x_star = np.array([5.0, 6.0, 7.0, 22.0, 35.0, 36.0, 37.0, 38.0])
    
    print("\nInitial Configuration:")
    for i in range(n):
        print(f"  Cell {i}: Width = {widths[i]}, Target X* = {x_star[i]}")
        
    # Solve Base Case (No D_max constraint)
    x_var = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(x_var - x_star))
    constraints = []
    c_left = x_var[0] >= llx
    constraints.append(c_left)
    c_spacing = []
    for i in range(n - 1):
        con = x_var[i+1] - x_var[i] >= widths[i]
        c_spacing.append(con)
        constraints.append(con)
    c_right = x_var[-1] + widths[-1] <= urx
    constraints.append(c_right)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    
    x_opt = x_var.value
    dual_left = c_left.dual_value
    dual_spacing = [c.dual_value for c in c_spacing]
    dual_right = c_right.dual_value
    
    print("\n=== Base QP Optimization Results ===")
    print(f"Problem Status: {prob.status}")
    print(f"Optimal Value (Displacement energy): {prob.value:.4f}")
    print("\nPrimal Solution:")
    for i in range(n):
        overlap_prev = ""
        if i > 0:
            gap = x_opt[i] - x_opt[i-1] - widths[i-1]
            overlap_prev = f" | Gap to prev: {gap:5.2f}"
        print(f"  Cell {i}: X* = {x_star[i]:4.1f} -> X_opt = {x_opt[i]:5.2f}{overlap_prev}")
        
    print("\nDual Solution (Legalization Pressure):")
    print(f"  Left Boundary Constraint (x[0] >= {llx:.1f}): Dual = {dual_left:.4f}")
    for i in range(n - 1):
        gap = x_opt[i+1] - x_opt[i] - widths[i]
        comp_slack = dual_spacing[i] * gap
        active_status = "ACTIVE (Touching)" if abs(gap) < 1e-4 else "INACTIVE (Separated)"
        print(f"  Spacing Constraint {i} -> {i+1} (gap >= {widths[i]:.1f}): "
              f"Actual Gap = {gap:5.2f} | Dual λ* = {dual_spacing[i]:7.4f} | λ* * gap = {comp_slack:10.2e} | {active_status}")
    print(f"  Right Boundary Constraint (x[-1] + {widths[-1]:.1f} <= {urx:.1f}): Dual = {dual_right:.4f}")
    
    # Save Dual Pressure Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1, 1.2]})
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896']
    
    # Pre-legalization
    ax = axes[0]
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(llx, color='black', linewidth=1.5, linestyle='-')
    ax.axvline(urx, color='black', linewidth=1.5, linestyle='-')
    for i in range(n):
        y_pos = i * 0.3
        rect = patches.Rectangle((x_star[i], y_pos), widths[i], 0.25, edgecolor='black', facecolor=colors[i], alpha=0.6)
        ax.add_patch(rect)
        ax.text(x_star[i] + widths[i]/2.0, y_pos + 0.12, f"C{i}", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_title("Pre-Legalization (Global Placement positions with overlaps)", fontsize=11, fontweight='bold')
    ax.set_xlim(llx - 2, urx + 2)
    ax.set_ylim(-0.2, n * 0.3 + 0.2)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Post-legalization
    ax = axes[1]
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(llx, color='black', linewidth=1.5, linestyle='-')
    ax.axvline(urx, color='black', linewidth=1.5, linestyle='-')
    for i in range(n):
        rect = patches.Rectangle((x_opt[i], 0.1), widths[i], 0.5, edgecolor='black', facecolor=colors[i], alpha=0.9)
        ax.add_patch(rect)
        ax.text(x_opt[i] + widths[i]/2.0, 0.35, f"C{i}", ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_title("Post-Legalization (Optimal Non-overlapping positions)", fontsize=11, fontweight='bold')
    ax.set_xlim(llx - 2, urx + 2)
    ax.set_ylim(0, 0.7)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Dual variables
    ax = axes[2]
    labels = ["Left Boundary"] + [f"C{i} ⟷ C{i+1}" for i in range(n-1)] + ["Right Boundary"]
    pressures = [dual_left] + dual_spacing + [dual_right]
    bars = ax.bar(labels, pressures, color='#d62728', alpha=0.85, edgecolor='black', width=0.5)
    for bar in bars:
        height = bar.get_height()
        if height > 0.001:
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title("Dual Legalization Pressure (Lagrange Multipliers λ*)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Pressure (Constraint Sensitivity)")
    ax.set_ylim(0, max(pressures)*1.2 if max(pressures)>0.1 else 1.0)
    plt.xticks(rotation=30, ha='right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.suptitle("ADMM Row Legalization: Dual Pressure Sensitivity Analysis", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("dual_pressure_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Sweep D_max (Infinity-norm Displacement Constraint)
    print("\n=== Sweep D_max (Infinity-norm Displacement Constraint) ===")
    dmax_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, np.inf]
    
    results = []
    print("| D_max | Solver Status | Primal Cost (Displacement) | Max Actual Displacement |")
    print("|---|---|---|---|")
    
    sweep_dmax = []
    sweep_cost = []
    
    for d in dmax_vals:
        status, val, sol = solve_legalization_qp_with_dmax(x_star, widths, llx, urx, d_max=d if d != np.inf else None)
        if status in ["optimal", "feasible"] and sol is not None:
            max_disp = np.max(np.abs(sol - x_star))
            cost_str = f"{val:.4f}"
            disp_str = f"{max_disp:.4f}"
            sweep_dmax.append(d if d != np.inf else 18.0) # plot inf as 18.0 for visual reference
            sweep_cost.append(val)
        else:
            cost_str = "INFEASIBLE"
            disp_str = "N/A"
        d_str = f"{d}" if d != np.inf else "∞"
        print(f"| {d_str:5} | {status:13} | {cost_str:26} | {disp_str:23} |")
        
    # Plot trade-off curve
    plt.figure(figsize=(7, 4.5))
    # Filter out np.inf for the line plot or label it specially
    plot_dmax = [d for d in sweep_dmax if d < 18.0]
    plot_cost = [sweep_cost[i] for i, d in enumerate(sweep_dmax) if d < 18.0]
    
    plt.plot(plot_dmax, plot_cost, marker='o', color='#1f77b4', linewidth=2, label="Feasible Region")
    
    # Draw infeasible region barrier
    plt.axvspan(0.0, 3.6666, color='#ff9896', alpha=0.3, label="Infeasible Region")
    plt.axvline(3.6667, color='#d62728', linestyle='--', linewidth=1.5)
    plt.text(2.0, (min(plot_cost) + max(plot_cost)) / 2.0, "INFEASIBLE\n(D_max too small)", ha='center', color='#d62728', fontweight='bold')
    
    # Plot inf value as a horizontal reference line
    inf_cost = sweep_cost[-1]
    plt.axhline(inf_cost, color='#2ca02c', linestyle=':', linewidth=1.5, label=f"Unconstrained (D_max=∞) Cost: {inf_cost:.2f}")
    
    plt.title("Legalization Cost vs. Maximum Displacement constraint ($D_{max}$)", fontsize=11, fontweight='bold')
    plt.xlabel("Maximum Allowed Displacement $D_{max}$ (Infinity-norm)")
    plt.ylabel("Primal Displacement Cost (Displacement Energy)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    output_sweep_name = "displacement_vs_dmax.png"
    plt.savefig(output_sweep_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nD_max trade-off visualization saved as {output_sweep_name}")

if __name__ == "__main__":
    main()
