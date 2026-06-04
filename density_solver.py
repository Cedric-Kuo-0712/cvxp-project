import numpy as np

class DensityGrid:
    def __init__(self, die_size, bin_size_x=20, bin_size_y=20):
        self.llx, self.lly, self.urx, self.ury = die_size
        self.width = self.urx - self.llx
        self.height = self.ury - self.lly
        
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        
        self.nx = int(np.ceil(self.width / bin_size_x))
        self.ny = int(np.ceil(self.height / bin_size_y))
        
        # Grid coordinates
        self.bin_centers_x = self.llx + (np.arange(self.nx) + 0.5) * bin_size_x
        self.bin_centers_y = self.lly + (np.arange(self.ny) + 0.5) * bin_size_y
        
    def compute_density_map(self, positions, sizes):
        """
        Compute the density map using a smooth quadratic B-spline kernel.
        positions: dict of instName -> (x, y)
        sizes: dict of instName -> (w, h)
        Returns:
            rho: 2D array of shape (nx, ny)
        """
        rho = np.zeros((self.nx, self.ny))
        
        # We can implement a simplified but smooth kernel for cell density.
        # For a cell at (cx, cy) with size (w, h), we compute its overlap with each bin.
        # To make it smooth, we use a bell-shaped function for distance:
        # e.g., quadratic B-spline.
        # B-spline kernel for 1D:
        #   b(d) = 1 - 2*d^2   if 0 <= d < 0.5
        #          2*(1-d)^2   if 0.5 <= d < 1.0
        #          0           otherwise
        # where d = |x - cx| / range
        
        for name, (cx, cy) in positions.items():
            w, h = sizes[name]
            cell_area = w * h
            
            # Influence range: cell size + 2 bin sizes
            range_x = w / 2.0 + 1.5 * self.bin_size_x
            range_y = h / 2.0 + 1.5 * self.bin_size_y
            
            # Find candidate bins
            min_bin_x = max(0, int((cx - range_x - self.llx) / self.bin_size_x))
            max_bin_x = min(self.nx - 1, int((cx + range_x - self.llx) / self.bin_size_x))
            
            min_bin_y = max(0, int((cy - range_y - self.lly) / self.bin_size_y))
            max_bin_y = min(self.ny - 1, int((cy + range_y - self.lly) / self.bin_size_y))
            
            # Distribute cell area to bins
            # To ensure sum of contributions equals cell_area, we normalize the weights
            bin_weights = {}
            total_weight = 0.0
            
            for bx in range(min_bin_x, max_bin_x + 1):
                bx_center = self.bin_centers_x[bx]
                dx = abs(cx - bx_center) / range_x
                if dx >= 1.0:
                    wx = 0.0
                elif dx < 0.5:
                    wx = 1.0 - 2.0 * (dx ** 2)
                else:
                    wx = 2.0 * ((1.0 - dx) ** 2)
                    
                for by in range(min_bin_y, max_bin_y + 1):
                    by_center = self.bin_centers_y[by]
                    dy = abs(cy - by_center) / range_y
                    if dy >= 1.0:
                        wy = 0.0
                    elif dy < 0.5:
                        wy = 1.0 - 2.0 * (dy ** 2)
                    else:
                        wy = 2.0 * ((1.0 - dy) ** 2)
                        
                    w_total = wx * wy
                    if w_total > 0:
                        bin_weights[(bx, by)] = w_total
                        total_weight += w_total
                        
            if total_weight > 0:
                for (bx, by), w_t in bin_weights.items():
                    # Density is area contribution divided by bin area
                    bin_area = self.bin_size_x * self.bin_size_y
                    rho[bx, by] += (w_t / total_weight) * cell_area / bin_area
                    
        return rho

    def solve_poisson_fft(self, rho, target_density):
        """
        Solve Poisson equation using 2D FFT:
            \nabla^2 \phi = \rho - \rho_target
        Returns:
            Ex: electric field x component, 2D array (nx, ny)
            Ey: electric field y component, 2D array (nx, ny)
            potential_energy: scalar sum of phi * (rho - rho_target)
        """
        # Define charge density distribution
        q = rho - target_density
        
        # 2D FFT of charge density
        q_hat = np.fft.fft2(q)
        
        # Frequency grid
        u = np.fft.fftfreq(self.nx)[:, None] # nx x 1
        v = np.fft.fftfreq(self.ny)[None, :] # 1 x ny
        
        # Eigenvalues of 5-point discrete Laplacian:
        # \lambda = 2 * (1 - cos(2*pi*u)) + 2 * (1 - cos(2*pi*v))
        # Note: u and v are normalized frequencies in [-0.5, 0.5]
        denom = 2.0 * (1.0 - np.cos(2.0 * np.pi * u)) + 2.0 * (1.0 - np.cos(2.0 * np.pi * v))
        
        # To avoid division by zero at DC component (0, 0)
        denom[0, 0] = 1.0
        
        # Solve for potential in frequency domain
        # \phi_hat = - q_hat / \lambda (with negative sign since we want repulsive force)
        phi_hat = q_hat / denom
        phi_hat[0, 0] = 0.0 # DC component has 0 potential
        
        # Calculate electric field in frequency domain
        # E_x = - \nabla_x \phi. In DFT domain, derivative is j * sin(2*pi*u) * \phi_hat
        # Repulsive force is + E_x, so:
        Ex_hat = -1j * np.sin(2.0 * np.pi * u) * phi_hat
        Ey_hat = -1j * np.sin(2.0 * np.pi * v) * phi_hat
        
        # Inverse FFT to get spatial potential and field
        phi = np.real(np.fft.ifft2(phi_hat))
        Ex = np.real(np.fft.ifft2(Ex_hat))
        Ey = np.real(np.fft.ifft2(Ey_hat))
        
        # Potential energy: 0.5 * sum(q * phi)
        potential_energy = 0.5 * np.sum(q * phi)
        
        return Ex, Ey, potential_energy

    def compute_density_forces(self, positions, sizes, Ex, Ey):
        """
        Compute repulsive forces on each cell by interpolation from the electric field.
        Returns:
            forces: dict of instName -> (fx, fy)
        """
        forces = {}
        
        for name, (cx, cy) in positions.items():
            w, h = sizes[name]
            cell_area = w * h
            
            range_x = w / 2.0 + 1.5 * self.bin_size_x
            range_y = h / 2.0 + 1.5 * self.bin_size_y
            
            min_bin_x = max(0, int((cx - range_x - self.llx) / self.bin_size_x))
            max_bin_x = min(self.nx - 1, int((cx + range_x - self.llx) / self.bin_size_x))
            min_bin_y = max(0, int((cy - range_y - self.lly) / self.bin_size_y))
            max_bin_y = min(self.ny - 1, int((cy + range_y - self.lly) / self.bin_size_y))
            
            fx = 0.0
            fy = 0.0
            total_weight = 0.0
            bin_weights = {}
            
            for bx in range(min_bin_x, max_bin_x + 1):
                bx_center = self.bin_centers_x[bx]
                dx = abs(cx - bx_center) / range_x
                if dx >= 1.0:
                    wx = 0.0
                elif dx < 0.5:
                    wx = 1.0 - 2.0 * (dx ** 2)
                else:
                    wx = 2.0 * ((1.0 - dx) ** 2)
                    
                for by in range(min_bin_y, max_bin_y + 1):
                    by_center = self.bin_centers_y[by]
                    dy = abs(cy - by_center) / range_y
                    if dy >= 1.0:
                        wy = 0.0
                    elif dy < 0.5:
                        wy = 1.0 - 2.0 * (dy ** 2)
                    else:
                        wy = 2.0 * ((1.0 - dy) ** 2)
                        
                    w_total = wx * wy
                    if w_total > 0:
                        bin_weights[(bx, by)] = w_total
                        total_weight += w_total
                        
            if total_weight > 0:
                for (bx, by), w_t in bin_weights.items():
                    norm_w = w_t / total_weight
                    fx += norm_w * Ex[bx, by]
                    fy += norm_w * Ey[bx, by]
                    
            # Scale force by cell area to match derivative of energy
            forces[name] = (fx * cell_area, fy * cell_area)
            
        return forces

if __name__ == "__main__":
    # Small test
    grid = DensityGrid((0, 0, 100, 100), bin_size_x=10, bin_size_y=10)
    positions = {
        'C1': (45.0, 45.0),
        'C2': (55.0, 55.0)
    }
    sizes = {
        'C1': (20, 20),
        'C2': (20, 20)
    }
    
    rho = grid.compute_density_map(positions, sizes)
    # average density target
    total_area = sum(w*h for w,h in sizes.values())
    die_area = 100 * 100
    target_density = total_area / die_area
    
    Ex, Ey, pot = grid.solve_poisson_fft(rho, target_density)
    forces = grid.compute_density_forces(positions, sizes, Ex, Ey)
    
    print("Density Map (sum):", np.sum(rho))
    print("Potential Energy:", pot)
    for name, f in forces.items():
        print(f"Force on {name}: {f}")
