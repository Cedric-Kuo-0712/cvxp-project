import numpy as np
import cvxpy as cp

def main():
    print("=== SDP Partitioning Bonus Experiment ===")
    
    # 1. Build Laplacian for Case 1
    # Instances: C1 (0), C2 (1), C3 (2), C4 (3), C5 (4), C6 (5), C7 (6), C8 (7)
    inst_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    n = len(inst_names)
    
    # Areas for Case 1 cells:
    # C1: MC1 (width 7, height 10) -> Area = 70
    # C2: MC3 (width 17, height 12) -> Area = 204
    # C3: MC3 (width 17, height 12) -> Area = 204
    # C4: MC2 (width 14, height 10) -> Area = 140
    # C5: MC2 (width 14, height 10) -> Area = 140
    # C6: MC3 (width 17, height 12) -> Area = 204
    # C7: MC2 (width 14, height 10) -> Area = 140
    # C8: MC1 (width 7, height 10) -> Area = 70
    areas = np.array([70.0, 204.0, 204.0, 140.0, 140.0, 204.0, 140.0, 70.0])
    
    # Nets in Case 1:
    # N1: C1, C2 (weight 1.0)
    # N2: C2, C3, C7 (weight 0.5)
    # N3: C2, C8 (weight 1.0)
    # N4: C3, C6, C7 (weight 0.5)
    # N5: C4, C6, C5 (weight 0.5)
    # N6: C4, C5 (weight 1.0)
    
    W = np.zeros((n, n))
    
    def add_clique(nodes):
        d = len(nodes)
        if d <= 1:
            return
        weight = 1.0 / (d - 1)
        for i in range(d):
            for j in range(i + 1, d):
                u, v = nodes[i], nodes[j]
                W[u, v] += weight
                W[v, u] += weight
                
    add_clique([0, 1])       # N1
    add_clique([1, 2, 6])    # N2
    add_clique([1, 7])       # N3
    add_clique([2, 5, 6])    # N4
    add_clique([3, 5, 4])    # N5
    add_clique([3, 4])       # N6
    
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    print("\nAdjacency Matrix W:")
    print(W)
    print("\nLaplacian Matrix L:")
    print(L)
    
    # 2. Spectral Partitioning (Fiedler Vector)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Find second smallest eigenvalue
    fiedler_idx = 1
    fiedler_val = eigenvalues[fiedler_idx]
    fiedler_vec = eigenvectors[:, fiedler_idx]
    
    # Round spectral partition
    # Let's check balance: we sort by fiedler vector and find optimal split
    # Since we want to balance areas: TopDieMaxUtil = 80%, BottomDieMaxUtil = 90%
    # Total area = 1172. Max Top Die area = 1172 * 0.8 = 937.6. Max Bottom Die area = 1172 * 0.9 = 1054.8
    # We can try all split indices and select the one that minimizes the cut size while satisfying utilization
    
    def calculate_cut_size(partition):
        # partition is list of -1 and 1
        cut = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if partition[i] != partition[j]:
                    cut += W[i, j]
        return cut

    best_spectral_cut = float('inf')
    best_spectral_partition = None
    
    sorted_indices = np.argsort(fiedler_vec)
    for split_idx in range(1, n):
        part = np.ones(n)
        part[sorted_indices[:split_idx]] = -1
        # Calculate areas
        top_area = np.sum(areas[part == 1])
        bot_area = np.sum(areas[part == -1])
        # Check utilization constraints
        if top_area <= 937.6 and bot_area <= 1054.8:
            cut = calculate_cut_size(part)
            if cut < best_spectral_cut:
                best_spectral_cut = cut
                best_spectral_partition = part
                
    print("\n=== Spectral Partitioning (Fiedler vector) ===")
    print(f"Fiedler Vector: {fiedler_vec}")
    print(f"Spectral Cut size: {best_spectral_cut:.2f}")
    spectral_top = [inst_names[i] for i in range(n) if best_spectral_partition[i] == 1]
    spectral_bot = [inst_names[i] for i in range(n) if best_spectral_partition[i] == -1]
    print(f"Top Die: {spectral_top} (Area = {np.sum(areas[best_spectral_partition == 1])})")
    print(f"Bottom Die: {spectral_bot} (Area = {np.sum(areas[best_spectral_partition == -1])})")

    # 3. Formulate and Solve SDP Partitioning
    print("\n=== Solving SDP Partitioning via CVXPY ===")
    X = cp.Variable((n, n), PSD=True)
    
    # Objective: Minimize 1/4 * Tr(L * X)
    objective = cp.Minimize(0.25 * cp.trace(L @ X))
    
    # Constraints:
    constraints = [
        cp.diag(X) == 1.0,
        # To enforce balance, we constrain the sum of matrix elements to be close to 0
        # s^T * 1 = 0 => 1^T * X * 1 = 0
        # Since we have different cell areas, we can constrain area-weighted sum:
        # a^T * X * a <= threshold
        # For simplicity, let's constrain the unweighted sum first, which represents number of cells balance:
        cp.sum(X) <= 2.0,  # allowing slight imbalance in number of cells
        cp.sum(X) >= -2.0
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    
    print(f"SDP Solver Status: {prob.status}")
    print(f"SDP Relaxation Lower Bound: {prob.value:.4f}")
    
    X_val = X.value
    print("\nSDP Solution Matrix X (rounded to 3 decimals):")
    print(np.round(X_val, 3))
    
    # 4. Rounding the SDP Solution
    # Method A: Eigenvector Rounding (first principal component)
    vals, vecs = np.linalg.eigh(X_val)
    v_max = vecs[:, -1] # eigenvector for largest eigenvalue
    sdp_part_eigen = np.sign(v_max)
    
    eigen_cut = calculate_cut_size(sdp_part_eigen)
    print("\nEigenvector Rounding Partition:")
    eigen_top = [inst_names[i] for i in range(n) if sdp_part_eigen[i] == 1]
    eigen_bot = [inst_names[i] for i in range(n) if sdp_part_eigen[i] == -1]
    print(f"  Cut size: {eigen_cut:.2f}")
    print(f"  Top Die: {eigen_top} (Area = {np.sum(areas[sdp_part_eigen == 1])})")
    print(f"  Bottom Die: {eigen_bot} (Area = {np.sum(areas[sdp_part_eigen == -1])})")
    
    # Method B: Goemans-Williamson Randomized Rounding (1000 trials)
    np.random.seed(42)
    best_rand_cut = float('inf')
    best_rand_part = None
    
    # Compute Cholesky decomposition X = V^T * V to get node vectors
    # Since X might have tiny negative eigenvalues due to numerical errors, we use eigh to compute the square root
    U, S, Vt = np.linalg.svd(X_val)
    V = U @ np.diag(np.sqrt(np.maximum(0.0, S)))
    
    for _ in range(1000):
        # Draw random vector r on unit sphere
        r = np.random.normal(size=n)
        r /= np.linalg.norm(r)
        
        # Round: s_i = sign(v_i^T * r)
        part = np.sign(V @ r)
        part[part == 0] = 1 # handle boundary
        
        # Check utilization constraints
        top_area = np.sum(areas[part == 1])
        bot_area = np.sum(areas[part == -1])
        if top_area <= 937.6 and bot_area <= 1054.8:
            cut = calculate_cut_size(part)
            if cut < best_rand_cut:
                best_rand_cut = cut
                best_rand_part = part
                
    print("\nGoemans-Williamson Randomized Rounding Partition:")
    if best_rand_part is not None:
        rand_top = [inst_names[i] for i in range(n) if best_rand_part[i] == 1]
        rand_bot = [inst_names[i] for i in range(n) if best_rand_part[i] == -1]
        print(f"  Cut size: {best_rand_cut:.2f}")
        print(f"  Top Die: {rand_top} (Area = {np.sum(areas[best_rand_part == 1])})")
        print(f"  Bottom Die: {rand_bot} (Area = {np.sum(areas[best_rand_part == -1])})")
    else:
        print("  No feasible partition found during randomized rounding.")

if __name__ == "__main__":
    main()
