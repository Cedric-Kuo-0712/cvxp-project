import numpy as np
# pyrefly: ignore [missing-import]
import scipy.sparse as sp
from parser import parse_input
from partitioner import spectral_partition
from qp_solver import construct_laplacian, run_irls_qp
from density_solver import DensityGrid

def compute_lse_gradient(data, assignment, target_die, x, y, inst_names, inst_to_idx, gamma=0.5):
    """
    Compute the gradient of the Log-Sum-Exp (LSE) smooth HPWL model.
    Using the max-subtraction stabilization trick to prevent float overflow.
    """
    n = len(inst_names)
    grad_wx = np.zeros(n)
    grad_wy = np.zeros(n)
    
    for net_name, net in data.nets.items():
        die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
        if len(die_pins) <= 1:
            continue
            
        idxs = [inst_to_idx[p[0]] for p in die_pins]
        
        # X-gradient
        xs = x[idxs]
        xmax = np.max(xs)
        xmin = np.min(xs)
        
        exp_pos = np.exp(gamma * (xs - xmax))
        exp_neg = np.exp(-gamma * (xs - xmin))
        
        sum_pos = np.sum(exp_pos)
        sum_neg = np.sum(exp_neg)
        
        # Y-gradient
        ys = y[idxs]
        ymax = np.max(ys)
        ymin = np.min(ys)
        
        exp_pos_y = np.exp(gamma * (ys - ymax))
        exp_neg_y = np.exp(-gamma * (ys - ymin))
        
        sum_pos_y = np.sum(exp_pos_y)
        sum_neg_y = np.sum(exp_neg_y)
        
        # Accumulate pin gradients
        for i, idx in enumerate(idxs):
            grad_wx[idx] += (exp_pos[i] / sum_pos - exp_neg[i] / sum_neg)
            grad_wy[idx] += (exp_pos_y[i] / sum_pos_y - exp_neg_y[i] / sum_neg_y)
            
    return grad_wx, grad_wy

def run_nesterov_placer(data, assignment, target_die, num_iterations=100, lr=0.1, init_lambda=0.01, density_solver='fft', check_kkt=False, wirelength_model='quadratic', gamma=0.5):
    """
    Run Nesterov Accelerated Gradient (NAG) global placement on a single die.
    Minimizes: f(x, y) = W_HPWL(x, y) + lambda * Potential_Energy(x, y)
    """
    inst_names = [name for name, die in assignment.items() if die == target_die]
    inst_to_idx = {name: i for i, name in enumerate(inst_names)}
    n = len(inst_names)
    
    if n == 0:
        return {}
        
    # Get technology cell library sizes for areas and dimensions
    tech_name = data.top_die_tech if target_die == 'top' else data.bottom_die_tech
    tech = data.technologies[tech_name]
    areas = []
    sizes = {}
    for name in inst_names:
        inst = data.instances[name]
        lc = tech.lib_cells[inst.lib_cell_name]
        areas.append(lc.size_x * lc.size_y)
        sizes[name] = (lc.size_x, lc.size_y)
    areas = np.array(areas, dtype=float)
    
    # Initialize with QP placement
    print(f"  [Nesterov Placer] Running initial QP on {target_die} die...")
    llx, lly, urx, ury = data.die_size
    width = urx - llx
    height = ury - lly
    center_x = (llx + urx) / 2.0
    center_y = (lly + ury) / 2.0
    qp_positions = run_irls_qp(data, assignment, target_die, num_iterations=3, center_x=center_x, center_y=center_y, check_kkt=check_kkt)
    
    x = np.array([qp_positions[name][0] for name in inst_names])
    y = np.array([qp_positions[name][1] for name in inst_names])
    
    # Add random perturbation to break symmetry and prevent exact overlaps
    np.random.seed(42)
    x = x + np.random.uniform(-0.02 * width, 0.02 * width, n)
    y = y + np.random.uniform(-0.02 * height, 0.02 * height, n)
    x = np.clip(x, llx, urx)
    y = np.clip(y, lly, ury)
    
    x_prev = x.copy()
    y_prev = y.copy()
    
    # Initialize density grid dynamically
    target_bins = 32
    bin_size_x = max(1.0, width / target_bins)
    bin_size_y = max(1.0, height / target_bins)
    grid = DensityGrid(data.die_size, bin_size_x=bin_size_x, bin_size_y=bin_size_y)
    
    # Target density: total cell area / total die area
    die_area = (urx - llx) * (ury - lly)
    total_cell_area = sum(areas)
    target_density = total_cell_area / die_area
    
    # Build initial Laplacian matrix for wirelength gradient calculation
    L = construct_laplacian(data, inst_names, inst_to_idx, assignment, target_die)
    
    lam = init_lambda
    
    print(f"  [Nesterov Placer] Target density = {target_density:.4f}, total cell area = {total_cell_area}")
    
    for k in range(num_iterations):
        # 1. Nesterov extrapolation step
        beta = k / (k + 3.0)
        yx = x + beta * (x - x_prev)
        yy = y + beta * (y - y_prev)
        
        # Keep within die boundaries
        yx = np.clip(yx, llx, urx)
        yy = np.clip(yy, lly, ury)
        
        # 2. Compute wirelength gradient: grad_w
        if wirelength_model == 'lse':
            grad_wx, grad_wy = compute_lse_gradient(data, assignment, target_die, yx, yy, inst_names, inst_to_idx, gamma)
        else:
            grad_wx = L.dot(yx)
            grad_wy = L.dot(yy)
        
        # 3. Compute density gradient (forces)
        curr_positions = {inst_names[i]: (yx[i], yy[i]) for i in range(n)}
        rho = grid.compute_density_map(curr_positions, sizes)
        if density_solver == 'dct':
            Ex, Ey, pot = grid.solve_poisson_dct(rho, target_density)
        else:
            Ex, Ey, pot = grid.solve_poisson_fft(rho, target_density)
        forces = grid.compute_density_forces(curr_positions, sizes, Ex, Ey)
        
        grad_dx = np.array([-forces[name][0] for name in inst_names])
        grad_dy = np.array([-forces[name][1] for name in inst_names])
        
        # Balance initial lambda dynamically at first iteration
        if k == 0:
            norm_w = np.sqrt(np.mean(grad_wx**2 + grad_wy**2))
            norm_d = np.sqrt(np.mean(grad_dx**2 + grad_dy**2))
            if norm_d > 1e-6:
                lam = max(init_lambda, 0.2 * (norm_w / norm_d))
                print(f"  [Nesterov Placer] Dynamically set initial Lambda = {lam:.6f} (norm_w={norm_w:.4f}, norm_d={norm_d:.4f})")
        
        # 4. Total gradient
        grad_x = grad_wx + lam * grad_dx
        grad_y = grad_wy + lam * grad_dy
        
        # Save previous x and y
        x_prev = x.copy()
        y_prev = y.copy()
        
        # 5. Update positions using Nesterov gradient descent
        # We can dynamically adjust learning rate based on gradient norms
        grad_norm = np.sqrt(np.mean(grad_x**2 + grad_y**2))
        if grad_norm > 1e-3:
            # simple step sizing: limit maximum movement per step dynamically (5% of max die dimension)
            max_move = 0.05 * max(width, height)
            step_scale = min(1.0, max_move / (lr * grad_norm))
            x = yx - lr * step_scale * grad_x
            y = yy - lr * step_scale * grad_y
        else:
            x = yx - lr * grad_x
            y = yy - lr * grad_y
            
        x = np.clip(x, llx, urx)
        y = np.clip(y, lly, ury)
        
        # 6. Penalty schedule: increase lambda to enforce non-overlap
        if k % 10 == 0:
            lam *= 1.2
            
        if (k + 1) % 20 == 0 or k == 0:

            # Calculate current metrics
            hpwl = 0.0
            for net_name, net in data.nets.items():
                die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
                if len(die_pins) <= 1:
                    continue
                coords_x = [x[inst_to_idx[p[0]]] for p in die_pins]
                coords_y = [y[inst_to_idx[p[0]]] for p in die_pins]
                hpwl += (max(coords_x) - min(coords_x)) + (max(coords_y) - min(coords_y))
                
            density_overflow = np.sum(np.maximum(0, rho - target_density))
            print(f"    Iter {k+1:3d}: HPWL = {hpwl:7.2f}, Pot Energy = {pot:8.2f}, Overflow = {density_overflow:7.2f}, Lambda = {lam:.4f}")
            
    # Build final positions dict
    final_pos = {inst_names[i]: (x[i], y[i]) for i in range(n)}
    return final_pos

if __name__ == "__main__":
    test_file = "toy_example.txt"
    parsed = parse_input(test_file)
    partition = spectral_partition(parsed)
    
    print("\nRunning Nesterov Placer on Top Die...")
    top_pos = run_nesterov_placer(parsed, partition, 'top', num_iterations=40, lr=0.5)
    print("Final positions (partial):")
    for name, pos in list(top_pos.items())[:4]:
        print(f"  {name}: {pos}")
