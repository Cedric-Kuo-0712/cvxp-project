import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from parser import parse_input

def spectral_partition(data):
    """
    Perform spectral partitioning on the parsed netlist to assign instances to top and bottom dies.
    Returns:
        dict: instName -> die ('top' or 'bottom')
    """
    inst_names = list(data.instances.keys())
    inst_to_idx = {name: i for i, name in enumerate(inst_names)}
    idx_to_inst = {i: name for i, name in enumerate(inst_names)}
    n = len(inst_names)
    
    if n <= 1:
        # Trivial case
        return {name: 'bottom' for name in inst_names}
        
    # Get technology cell library sizes for utilization calculation
    # We will assume each cell is placed on its target technology.
    # But since partition happens BEFORE we know which die it goes to, and the cell sizes can differ 
    # between top and bottom tech, we need to handle this.
    # Let's estimate cell area. For partitioning, we can average the cell area on top and bottom tech,
    # or just use the bottom tech size as a representative since they are equivalent logic.
    # The problem description says: "both given technologies would have the same logical library cells by library cell name matching."
    top_tech_name = data.top_die_tech
    bottom_tech_name = data.bottom_die_tech
    
    top_tech = data.technologies[top_tech_name]
    bottom_tech = data.technologies[bottom_tech_name]
    
    # Calculate cell areas on top and bottom dies
    inst_areas_top = []
    inst_areas_bottom = []
    for name in inst_names:
        inst = data.instances[name]
        lc_top = top_tech.lib_cells[inst.lib_cell_name]
        lc_bottom = bottom_tech.lib_cells[inst.lib_cell_name]
        inst_areas_top.append(lc_top.size_x * lc_top.size_y)
        inst_areas_bottom.append(lc_bottom.size_x * lc_bottom.size_y)
        
    inst_areas_top = np.array(inst_areas_top)
    inst_areas_bottom = np.array(inst_areas_bottom)
    
    # Construct adjacency matrix W using clique expansion
    # For net with d pins, edge weight between its instances is 1 / (d - 1)
    # We will build a coordinate format sparse matrix
    row_indices = []
    col_indices = []
    weights = []
    
    for net_name, net in data.nets.items():
        pins = net.pins
        d = len(pins)
        if d <= 1:
            continue
        weight = 1.0 / (d - 1)
        # Unique instances connected by this net
        connected_insts = list(set([inst_name for inst_name, _ in pins if inst_name in inst_to_idx]))
        
        for i in range(len(connected_insts)):
            for j in range(i + 1, len(connected_insts)):
                u = inst_to_idx[connected_insts[i]]
                v = inst_to_idx[connected_insts[j]]
                # add symmetric edges
                row_indices.append(u)
                col_indices.append(v)
                weights.append(weight)
                
                row_indices.append(v)
                col_indices.append(u)
                weights.append(weight)
                
    # Build sparse matrix W
    if len(weights) > 0:
        W = sp.coo_matrix((weights, (row_indices, col_indices)), shape=(n, n)).tocsr()
    else:
        W = sp.csr_matrix((n, n))
        
    # Degree matrix D
    degrees = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    
    # Laplacian L = D - W
    L = D - W
    
    # Solve for the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)
    # We use shift-invert mode to find the smallest eigenvalues
    # Since L is symmetric and positive semidefinite, its smallest eigenvalue is 0 with eigenvector 1.
    # The second smallest eigenvalue corresponds to Fiedler vector.
    # To avoid solver issues with singular matrices, we can add a small shift or use spla.eigsh with 'SM' (smallest magnitude)
    try:
        # We ask for k=2 eigenvalues. Fiedler vector is eigenvectors[:, 1] (or [:, 0] depending on order)
        # We need a stable solver.
        val, vec = spla.eigsh(L.astype(float), k=min(n - 1, 2), which='SM')
        # Sort eigenvalues to be sure
        idx_sort = np.argsort(val)
        val = val[idx_sort]
        vec = vec[:, idx_sort]
        
        if len(val) >= 2:
            fiedler = vec[:, 1]
        else:
            fiedler = vec[:, 0]
    except Exception as e:
        print(f"Eigen solver failed: {e}. Falling back to degree-based ordering.")
        # Fallback to degree or random
        fiedler = degrees + np.random.normal(0, 1e-4, n)
        
    # Partition based on Fiedler vector values
    # We want to assign elements to 'top' or 'bottom'
    # Sort indices by Fiedler vector values
    sorted_indices = np.argsort(fiedler)
    
    # Calculate Die area limits
    llx, lly, urx, ury = data.die_size
    die_area = (urx - llx) * (ury - lly)
    
    max_area_top = die_area * (data.top_die_max_util / 100.0)
    max_area_bottom = die_area * (data.bottom_die_max_util / 100.0)
    
    # We will find a cutoff to balance the area
    # Let's search for a split point 'p' such that:
    # instances with sorted_indices[0..p-1] are assigned to bottom die
    # instances with sorted_indices[p..n-1] are assigned to top die
    # (or vice versa, we should check which assignment is better or satisfies both constraints)
    
    best_p = -1
    best_assignment = None
    min_violation = float('inf')
    
    # Try all possible split points
    # Since n can be large, we can sample split points or do binary search,
    # but for small/medium cases we can scan.
    # To be efficient, we can precalculate cumulative sums.
    
    # Let's compute prefix sums of areas
    # bottom_area if sorted_indices[0..i] are bottom, and remaining are top
    areas_bottom_sorted = inst_areas_bottom[sorted_indices]
    areas_top_sorted = inst_areas_top[sorted_indices]
    
    pref_bottom = np.cumsum(areas_bottom_sorted)
    # suffix sum of top area
    suff_top = np.cumsum(areas_top_sorted[::-1])[::-1]
    
    for p in range(0, n + 1):
        # p is the number of elements in the bottom die
        if p == 0:
            area_b = 0
            area_t = suff_top[0]
        elif p == n:
            area_b = pref_bottom[-1]
            area_t = 0
        else:
            area_b = pref_bottom[p-1]
            area_t = suff_top[p]
            
        violation_b = max(0.0, area_b - max_area_bottom)
        violation_t = max(0.0, area_t - max_area_top)
        total_violation = violation_b + violation_t
        
        if total_violation < min_violation:
            min_violation = total_violation
            best_p = p
            best_assignment = 'bottom_first'  # [0..p-1] is bottom, [p..n-1] is top
            
    # Also try the opposite assignment (just in case)
    # [0..p-1] is top, [p..n-1] is bottom
    pref_top = np.cumsum(areas_top_sorted)
    suff_bottom = np.cumsum(areas_bottom_sorted[::-1])[::-1]
    
    for p in range(0, n + 1):
        if p == 0:
            area_t = 0
            area_b = suff_bottom[0]
        elif p == n:
            area_t = pref_top[-1]
            area_b = 0
        else:
            area_t = pref_top[p-1]
            area_b = suff_bottom[p]
            
        violation_b = max(0.0, area_b - max_area_bottom)
        violation_t = max(0.0, area_t - max_area_top)
        total_violation = violation_b + violation_t
        
        if total_violation < min_violation:
            min_violation = total_violation
            best_p = p
            best_assignment = 'top_first'  # [0..p-1] is top, [p..n-1] is bottom

    # Build the final assignment dict
    assignment = {}
    if best_assignment == 'bottom_first':
        for i, idx in enumerate(sorted_indices):
            inst_name = idx_to_inst[idx]
            if i < best_p:
                assignment[inst_name] = 'bottom'
            else:
                assignment[inst_name] = 'top'
    else:
        for i, idx in enumerate(sorted_indices):
            inst_name = idx_to_inst[idx]
            if i < best_p:
                assignment[inst_name] = 'top'
            else:
                assignment[inst_name] = 'bottom'
                
    # Calculate final utilization
    area_t = sum(inst_areas_top[inst_to_idx[name]] for name, die in assignment.items() if die == 'top')
    area_b = sum(inst_areas_bottom[inst_to_idx[name]] for name, die in assignment.items() if die == 'bottom')
    util_t = (area_t / die_area) * 100.0
    util_b = (area_b / die_area) * 100.0
    
    print(f"Partitioning Result:")
    print(f"  Top Die Area Util: {util_t:.2f}% (Max: {data.top_die_max_util}%)")
    print(f"  Bottom Die Area Util: {util_b:.2f}% (Max: {data.bottom_die_max_util}%)")
    print(f"  Total violation: {min_violation:.2f}")
    
    # If there is still a violation, we can perform a simple greedy refinement
    # to move cells between top and bottom to resolve utilization constraints.
    if min_violation > 0:
        print("  Warning: Initial spectral partition violated utilization constraints. Refining...")
        # Greedy swap or moves:
        # Move cells from the violating die to the non-violating die.
        # We sort cells by their Fiedler vector value proximity to the cutoff, so we move cells that are
        # most "neutral" (close to the boundary).
        # Let's say top die is violating. We want to move cells from top to bottom.
        # Which cells? The ones in top die that are closest to the cutoff index in the sorted array.
        # Let's recompute areas
        while True:
            # Recompute areas
            area_t = sum(inst_areas_top[inst_to_idx[name]] for name, die in assignment.items() if die == 'top')
            area_b = sum(inst_areas_bottom[inst_to_idx[name]] for name, die in assignment.items() if die == 'bottom')
            
            if area_t > max_area_top and area_b <= max_area_bottom:
                # Move from top to bottom
                # Find cell in top die with index closest to best_p in sorted_indices
                candidates = []
                for i, idx in enumerate(sorted_indices):
                    inst_name = idx_to_inst[idx]
                    if assignment[inst_name] == 'top':
                        candidates.append((abs(i - best_p), inst_name))
                candidates.sort()
                if candidates:
                    to_move = candidates[0][1]
                    # Check if moving this cell doesn't violate bottom too much or we have no choice
                    assignment[to_move] = 'bottom'
                    continue
            elif area_b > max_area_bottom and area_t <= max_area_top:
                # Move from bottom to top
                candidates = []
                for i, idx in enumerate(sorted_indices):
                    inst_name = idx_to_inst[idx]
                    if assignment[inst_name] == 'bottom':
                        candidates.append((abs(i - best_p), inst_name))
                candidates.sort()
                if candidates:
                    to_move = candidates[0][1]
                    assignment[to_move] = 'top'
                    continue
            break
            
        area_t = sum(inst_areas_top[inst_to_idx[name]] for name, die in assignment.items() if die == 'top')
        area_b = sum(inst_areas_bottom[inst_to_idx[name]] for name, die in assignment.items() if die == 'bottom')
        util_t = (area_t / die_area) * 100.0
        util_b = (area_b / die_area) * 100.0
        print(f"Refined Partitioning Result:")
        print(f"  Top Die Area Util: {util_t:.2f}% (Max: {data.top_die_max_util}%)")
        print(f"  Bottom Die Area Util: {util_b:.2f}% (Max: {data.bottom_die_max_util}%)")
        
    return assignment

if __name__ == "__main__":
    test_file = "toy_example.txt"
    parsed = parse_input(test_file)
    partition = spectral_partition(parsed)
    print("Instance Assignment:")
    for inst, die in partition.items():
        print(f"  {inst}: {die}")
