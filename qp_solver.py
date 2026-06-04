import numpy as np
# pyrefly: ignore [missing-import]
import scipy.sparse as sp
# pyrefly: ignore [missing-import]
import scipy.sparse.linalg as spla

def solve_constrained_qp(L, areas, target_center, epsilon=1e-5, check_kkt=False):
    """
    Solve the equality-constrained QP:
        min  1/2 * x^T * L * x
        s.t. areas^T * x = target_center * sum(areas)
    
    Using Schur complement:
        Let L_reg = L + epsilon * I  (to make it strictly positive definite)
        We want to solve:
            L_reg * x + areas * lambda = 0
            areas^T * x = C
        From (1): x = - L_reg^-1 * areas * lambda
        Substitute into (2): - areas^T * L_reg^-1 * areas * lambda = C
        Therefore:
            lambda = - C / (areas^T * L_reg^-1 * areas)
            x = - L_reg^-1 * areas * lambda
    """
    n = L.shape[0]
    areas = np.array(areas, dtype=float)
    C = target_center * np.sum(areas)
    
    # Regularize L to make it SPD
    L_reg = L + sp.diags([epsilon] * n)
    
    # Solve L_reg * v = areas
    # Since L_reg is symmetric positive definite, we can use CG or spsolve
    # CG is iterative and suitable for large sparse systems
    v, info = spla.cg(L_reg, areas, rtol=1e-5, maxiter=1000)
    if info != 0:
        # Fallback to direct solver if CG has issues
        v = spla.spsolve(L_reg, areas)
        
    denom = np.dot(areas, v)
    if abs(denom) < 1e-9:
        # Avoid division by zero
        denom = 1e-9
        
    lam = -C / denom
    x = -v * lam
    
    if check_kkt:
        primal_feas_val = np.dot(areas, x)
        primal_residual = abs(primal_feas_val - C)
        
        stationarity_vec = L_reg.dot(x) + areas * lam
        stationarity_residual = np.linalg.norm(stationarity_vec)
        
        norm_L_reg_x = np.linalg.norm(L_reg.dot(x))
        norm_areas_lam = np.linalg.norm(areas * lam)
        scale = max(1e-9, norm_L_reg_x, norm_areas_lam)
        rel_stationarity = stationarity_residual / scale
        
        print(f"      [KKT Check] Primal Feasibility Residual: {primal_residual:.2e} (C = {C:.2e})")
        print(f"      [KKT Check] Stationarity Residual: {stationarity_residual:.2e} (Relative: {rel_stationarity:.2e})")
        
    return x

def construct_laplacian(data, inst_names, inst_to_idx, assignment, target_die, weights=None):
    """
    Construct the Laplacian matrix for a specific die.
    Also returns fixed vector b if there are fixed points (like terminals).
    For now, we build the unnormalized Laplacian.
    """
    n = len(inst_names)
    row_indices = []
    col_indices = []
    vals = []
    
    # If weights is not provided, initialize with 1 / (d - 1)
    if weights is None:
        weights = {}
        for net_name, net in data.nets.items():
            d = len(net.pins)
            if d <= 1:
                continue
            weights[net_name] = 1.0 / (d - 1)
            
    for net_name, net in data.nets.items():
        pins = net.pins
        # Filter pins that belong to the target die
        die_pins = [p for p in pins if assignment.get(p[0]) == target_die]
        d = len(die_pins)
        if d <= 1:
            continue
            
        w = weights.get(net_name, 1.0 / (len(pins) - 1))
        
        for i in range(d):
            for j in range(i + 1, d):
                u = inst_to_idx[die_pins[i][0]]
                v = inst_to_idx[die_pins[j][0]]
                
                row_indices.append(u)
                col_indices.append(v)
                vals.append(-w)
                
                row_indices.append(v)
                col_indices.append(u)
                vals.append(-w)
                
    # Build W matrix
    if len(vals) > 0:
        W = sp.coo_matrix((vals, (row_indices, col_indices)), shape=(n, n)).tocsr()
    else:
        W = sp.csr_matrix((n, n))
        
    # Degree matrix D
    degrees = np.array(-W.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    
    L = D + W  # Since W has negative off-diagonal entries, L = D - (-W) = D + W
    return L

def run_irls_qp(data, assignment, target_die, num_iterations=5, center_x=250.0, center_y=225.0, check_kkt=False):
    """
    Run Iteratively Reweighted Least Squares (IRLS) on a single die.
    Approximates L1 HPWL.
    """
    inst_names = [name for name, die in assignment.items() if die == target_die]
    inst_to_idx = {name: i for i, name in enumerate(inst_names)}
    n = len(inst_names)
    
    if n == 0:
        return {}
        
    # Get technology cell library sizes for areas
    tech_name = data.top_die_tech if target_die == 'top' else data.bottom_die_tech
    tech = data.technologies[tech_name]
    areas = []
    for name in inst_names:
        inst = data.instances[name]
        lc = tech.lib_cells[inst.lib_cell_name]
        areas.append(lc.size_x * lc.size_y)
    areas = np.array(areas, dtype=float)
    
    # Initialize positions (all at center)
    x = np.full(n, center_x)
    y = np.full(n, center_y)
    
    # Initialize weights
    weights = {}
    for net_name, net in data.nets.items():
        d = len(net.pins)
        if d <= 1:
            continue
        weights[net_name] = 1.0 / (d - 1)
        
    for k in range(num_iterations):
        # Construct Laplacian for x and y
        Lx = construct_laplacian(data, inst_names, inst_to_idx, assignment, target_die, weights)
        Ly = Lx  # Topology is same, so Ly is same as Lx
        
        # Solve constrained QP
        x_new = solve_constrained_qp(Lx, areas, center_x, check_kkt=check_kkt)
        y_new = solve_constrained_qp(Ly, areas, center_y, check_kkt=check_kkt)
        
        # Update weights based on current distance (IRLS)
        # For a net, the "span" is max(coord) - min(coord).
        # We can update the net weight by 1 / (span + delta)
        delta = 1.0  # to avoid division by zero
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
            if len(die_pins) <= 1:
                continue
            coords_x = [x_new[inst_to_idx[p[0]]] for p in die_pins]
            coords_y = [y_new[inst_to_idx[p[0]]] for p in die_pins]
            
            span_x = max(coords_x) - min(coords_x)
            span_y = max(coords_y) - min(coords_y)
            span = max(1.0, span_x + span_y)
            
            # Update weights: standard IRLS formula
            weights[net_name] = 1.0 / (span + delta)
            
        x = x_new
        y = y_new
        
        # Calculate HPWL of this die
        hpwl = 0.0
        for net_name, net in data.nets.items():
            die_pins = [p for p in net.pins if assignment.get(p[0]) == target_die]
            if len(die_pins) <= 1:
                continue
            coords_x = [x[inst_to_idx[p[0]]] for p in die_pins]
            coords_y = [y[inst_to_idx[p[0]]] for p in die_pins]
            hpwl += (max(coords_x) - min(coords_x)) + (max(coords_y) - min(coords_y))
            
        # print(f"  IRLS Iteration {k+1}: HPWL = {hpwl:.2f}")
        
    return {name: (x[inst_to_idx[name]], y[inst_to_idx[name]]) for name in inst_names}

if __name__ == "__main__":
    from parser import parse_input
    from partitioner import spectral_partition
    
    test_file = "toy_example.txt"
    parsed = parse_input(test_file)
    partition = spectral_partition(parsed)
    
    print("\nRunning QP Placer on Top Die...")
    top_pos = run_irls_qp(parsed, partition, 'top', num_iterations=3)
    for name, pos in top_pos.items():
        print(f"  {name}: {pos}")
        
    print("\nRunning QP Placer on Bottom Die...")
    bottom_pos = run_irls_qp(parsed, partition, 'bottom', num_iterations=3)
    for name, pos in bottom_pos.items():
        print(f"  {name}: {pos}")
