import numpy as np
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
from parser import parse_input
from partitioner import spectral_partition
from qp_solver import construct_laplacian, solve_constrained_qp
from density_solver import DensityGrid

def run_gd_vs_nesterov(data, assignment, target_die='top', num_iterations=100, lr=0.5, lam=0.01):
    """
    Compare standard Gradient Descent vs Nesterov Accelerated Gradient.
    Returns:
        gd_history: list of (hpwl, potential_energy, overflow)
        nag_history: list of (hpwl, potential_energy, overflow)
    """
    inst_names = [name for name, die in assignment.items() if die == target_die]
    inst_to_idx = {name: i for i, name in enumerate(inst_names)}
    n = len(inst_names)
    
    tech_name = data.top_die_tech if target_die == 'top' else data.bottom_die_tech
    tech = data.technologies[tech_name]
    sizes = {name: (tech.lib_cells[data.instances[name].lib_cell_name].size_x,
                    tech.lib_cells[data.instances[name].lib_cell_name].size_y)
             for name in inst_names}
             
    # Target density
    llx, lly, urx, ury = data.die_size
    die_area = (urx - llx) * (ury - lly)
    total_area = sum(sizes[name][0]*sizes[name][1] for name in inst_names)
    target_density = total_area / die_area
    grid = DensityGrid(data.die_size, bin_size_x=20, bin_size_y=20)
    L = construct_laplacian(data, inst_names, inst_to_idx, assignment, target_die)
    
    # 1. Run Standard Gradient Descent
    x = np.full(n, (llx + urx)/2.0) + np.random.uniform(-5.0, 5.0, n)
    y = np.full(n, (lly + ury)/2.0) + np.random.uniform(-5.0, 5.0, n)
    gd_history = []
    
    for k in range(num_iterations):
        grad_wx = L.dot(x)
        grad_wy = L.dot(y)
        
        curr_pos = {inst_names[i]: (x[i], y[i]) for i in range(n)}
        rho = grid.compute_density_map(curr_pos, sizes)
        Ex, Ey, pot = grid.solve_poisson_fft(rho, target_density)
        forces = grid.compute_density_forces(curr_pos, sizes, Ex, Ey)
        
        grad_dx = np.array([-forces[name][0] for name in inst_names])
        grad_dy = np.array([-forces[name][1] for name in inst_names])
        
        grad_x = grad_wx + lam * grad_dx
        grad_y = grad_wy + lam * grad_dy
        
        # Calculate HPWL
        hpwl = 0.0
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
            if len(die_pins) <= 1:
                continue
            coords_x = [x[inst_to_idx[p[0]]] for p in die_pins]
            coords_y = [y[inst_to_idx[p[0]]] for p in die_pins]
            hpwl += (max(coords_x) - min(coords_x)) + (max(coords_y) - min(coords_y))
            
        overflow = np.sum(np.maximum(0, rho - target_density))
        gd_history.append((hpwl, pot, overflow))
        
        x = np.clip(x - lr * grad_x, llx, urx)
        y = np.clip(y - lr * grad_y, lly, ury)
        
    # 2. Run Nesterov AGD
    x = np.full(n, (llx + urx)/2.0) + np.random.uniform(-5.0, 5.0, n)
    y = np.full(n, (lly + ury)/2.0) + np.random.uniform(-5.0, 5.0, n)
    x_prev = x.copy()
    y_prev = y.copy()
    nag_history = []
    
    for k in range(num_iterations):
        beta = k / (k + 3.0)
        yx = x + beta * (x - x_prev)
        yy = y + beta * (y - y_prev)
        
        grad_wx = L.dot(yx)
        grad_wy = L.dot(yy)
        
        curr_pos = {inst_names[i]: (yx[i], yy[i]) for i in range(n)}
        rho = grid.compute_density_map(curr_pos, sizes)
        Ex, Ey, pot = grid.solve_poisson_fft(rho, target_density)
        forces = grid.compute_density_forces(curr_pos, sizes, Ex, Ey)
        
        grad_dx = np.array([-forces[name][0] for name in inst_names])
        grad_dy = np.array([-forces[name][1] for name in inst_names])
        
        grad_x = grad_wx + lam * grad_dx
        grad_y = grad_wy + lam * grad_dy
        
        hpwl = 0.0
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
            if len(die_pins) <= 1:
                continue
            coords_x = [yx[inst_to_idx[p[0]]] for p in die_pins]
            coords_y = [yy[inst_to_idx[p[0]]] for p in die_pins]
            hpwl += (max(coords_x) - min(coords_x)) + (max(coords_y) - min(coords_y))
            
        overflow = np.sum(np.maximum(0, rho - target_density))
        nag_history.append((hpwl, pot, overflow))
        
        x_prev = x.copy()
        y_prev = y.copy()
        
        x = np.clip(yx - lr * grad_x, llx, urx)
        y = np.clip(yy - lr * grad_y, lly, ury)
        
    return gd_history, nag_history

def generate_report_plots():
    test_file = "testcase_official/ProblemB_case1_0522.txt"
    if not os.path.exists(test_file):
        test_file = "toy_example.txt"
    print(f"Using test file: {test_file}")
    parsed = parse_input(test_file)
    partition = spectral_partition(parsed)
    
    print("Running optimization comparison...")
    gd_hist, nag_hist = run_gd_vs_nesterov(parsed, partition, target_die='top', num_iterations=50, lr=0.5, lam=0.05)
    
    # Extract data
    gd_hpwl = [h[0] for h in gd_hist]
    gd_pot = [h[1] for h in gd_hist]
    gd_over = [h[2] for h in gd_hist]
    
    nag_hpwl = [h[0] for h in nag_hist]
    nag_pot = [h[1] for h in nag_hist]
    nag_over = [h[2] for h in nag_hist]
    
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. HPWL
    axs[0].plot(gd_hpwl, label='Gradient Descent', color='red', linestyle='--')
    axs[0].plot(nag_hpwl, label='Nesterov AGD', color='blue')
    axs[0].set_title('HPWL Convergence')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('HPWL')
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. Potential Energy
    axs[1].plot(gd_pot, label='Gradient Descent', color='red', linestyle='--')
    axs[1].plot(nag_pot, label='Nesterov AGD', color='blue')
    axs[1].set_title('Potential Energy (Density Penalty)')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Energy')
    axs[1].legend()
    axs[1].grid(True)
    
    # 3. Density Overflow
    axs[2].plot(gd_over, label='Gradient Descent', color='red', linestyle='--')
    axs[2].plot(nag_over, label='Nesterov AGD', color='blue')
    axs[2].set_title('Density Overflow')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Overflow')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plot_path = "convergence_comparison.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved successfully to {plot_path}!")

if __name__ == "__main__":
    try:
        generate_report_plots()
    except Exception as e:
        print(f"Plotting failed (maybe missing matplotlib): {e}")
