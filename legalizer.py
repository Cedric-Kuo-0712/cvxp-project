import numpy as np

def optimize_and_legalize_terminals(data, assignment, cell_positions):
    """
    1. Identify crossing nets.
    2. Compute the geometric median (x, y) of the pins for each crossing net.
    3. Legalize terminal positions to satisfy size, spacing, and boundary constraints.
    Returns:
        dict: net_name -> (tx, ty) (center coordinates of the terminal)
    """
    # 1. Find crossing nets
    crossing_nets = []
    for net_name, net in data.nets.items():
        dies_connected = set()
        for inst_name, _ in net.pins:
            if inst_name in assignment:
                dies_connected.add(assignment[inst_name])
        if len(dies_connected) > 1:
            crossing_nets.append(net_name)
            
    if not crossing_nets:
        return {}
        
    # Die dimensions
    llx, lly, urx, ury = data.die_size
    die_w = urx - llx
    die_h = ury - lly
    
    # Terminal constraints
    tw, th = data.terminal_size
    ts = data.terminal_spacing
    
    # Allowed region for terminal center (tx, ty)
    min_tx = llx + ts + tw / 2.0
    max_tx = urx - ts - tw / 2.0
    min_ty = lly + ts + th / 2.0
    max_ty = ury - ts - th / 2.0
    
    # 2. Compute ideal terminal positions (median of connected pins)
    ideal_terminals = {}
    for net_name in crossing_nets:
        net = data.nets[net_name]
        pin_xs = []
        pin_ys = []
        for inst_name, pin_name in net.pins:
            # We need to get the pin offset relative to cell origin
            inst = data.instances[inst_name]
            die = assignment[inst_name]
            tech_name = data.top_die_tech if die == 'top' else data.bottom_die_tech
            tech = data.technologies[tech_name]
            lc = tech.lib_cells[inst.lib_cell_name]
            
            # Global cell position
            cx, cy = cell_positions[inst_name]
            
            # Pin local position (offset)
            px_off, py_off = lc.pins.get(pin_name, (0, 0))
            
            pin_xs.append(cx + px_off)
            pin_ys.append(cy + py_off)
            
        # Geometric median is the 1D median for x and y
        tx = np.median(pin_xs)
        ty = np.median(pin_ys)
        
        # Clamp to allowed boundary
        tx = np.clip(tx, min_tx, max_tx)
        ty = np.clip(ty, min_ty, max_ty)
        
        ideal_terminals[net_name] = (tx, ty)
        
    # 3. Legalize terminals using a grid-based matching to resolve spacing conflicts.
    # Grid spacing is tw + ts
    grid_step_x = tw + ts
    grid_step_y = th + ts
    
    # Construct a list of valid grid points
    grid_xs = np.arange(min_tx, max_tx + 1, grid_step_x)
    grid_ys = np.arange(min_ty, max_ty + 1, grid_step_y)
    
    # Create grid points
    grid_points = []
    for gx in grid_xs:
        for gy in grid_ys:
            grid_points.append((gx, gy))
            
    # Solve matching: for each terminal, find the closest available grid point
    # We sort terminals by their ideal coordinates or just use a simple greedy approach.
    legalized_terminals = {}
    used_grid_indices = set()
    
    # Sort nets by how constrained they are or simply by their name to be deterministic
    sorted_nets = sorted(crossing_nets)
    
    for net_name in sorted_nets:
        tx_ideal, ty_ideal = ideal_terminals[net_name]
        
        # Find closest unused grid point
        best_grid_idx = -1
        min_dist = float('inf')
        
        for idx, (gx, gy) in enumerate(grid_points):
            if idx in used_grid_indices:
                continue
            dist = (gx - tx_ideal)**2 + (gy - ty_ideal)**2
            if dist < min_dist:
                min_dist = dist
                best_grid_idx = idx
                
        if best_grid_idx != -1:
            used_grid_indices.add(best_grid_idx)
            gx, gy = grid_points[best_grid_idx]
            # Convert center to integer as required by the contest rules
            legalized_terminals[net_name] = (int(np.round(gx)), int(np.round(gy)))
        else:
            # Fallback if grid runs out of points (very crowded die, shouldn't happen for toy/normal cases)
            print(f"Warning: Out of terminal grid points for net {net_name}!")
            legalized_terminals[net_name] = (int(np.round(tx_ideal)), int(np.round(ty_ideal)))
            
    return legalized_terminals

def resolve_macro_overlaps(macros, sizes, positions, die_size):
    llx, lly, urx, ury = die_size
    placed = {}
    
    # Sort macros by area descending
    sorted_macros = sorted(macros, key=lambda m: (-sizes[m][0]*sizes[m][1], m))
    
    # If no macros, return empty
    if not sorted_macros:
        return {}
        
    call_count = 0
    max_calls = 3000  # Prevent deep recursion on extreme test cases
    
    # Spiral search step size dynamically scaled to die size
    step_x = max(1, (urx - llx) // 60)
    step_y = max(1, (ury - lly) // 60)
    max_steps = int(max(urx - llx, ury - lly) / min(step_x, step_y)) + 2
    
    def search(idx):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            return False
            
        if idx == len(sorted_macros):
            return True
            
        m = sorted_macros[idx]
        mx, my = positions[m]
        mw, mh = sizes[m]
        
        # Initial placement clamped to die
        mx_clamped = int(np.clip(np.round(mx), llx, urx - mw))
        my_clamped = int(np.clip(np.round(my), lly, ury - mh))
        
        def overlaps_any(x, y):
            for pm in sorted_macros[:idx]:
                px, py = placed[pm]
                pw, ph = sizes[pm]
                if max(x, px) < min(x + mw, px + pw) and max(y, py) < min(y + mh, py + ph):
                    return True
            return False
            
        # 1. Try original position first
        if not overlaps_any(mx_clamped, my_clamped):
            placed[m] = (mx_clamped, my_clamped)
            if search(idx + 1):
                return True
                
        # 2. Try nearby spiral coordinates
        for s in range(1, max_steps):
            candidates = []
            for ds in range(-s, s + 1):
                for dy in [-s, s]:
                    tx = int(np.clip(mx_clamped + ds * step_x, llx, urx - mw))
                    ty = int(np.clip(my_clamped + dy * step_y, lly, ury - mh))
                    if not overlaps_any(tx, ty):
                        candidates.append((tx, ty))
            for dy in range(-s + 1, s):
                for ds in [-s, s]:
                    tx = int(np.clip(mx_clamped + ds * step_x, llx, urx - mw))
                    ty = int(np.clip(my_clamped + dy * step_y, lly, ury - mh))
                    if not overlaps_any(tx, ty):
                        candidates.append((tx, ty))
                        
            # Sort candidates by distance to original target positions
            candidates = sorted(list(set(candidates)), key=lambda pos: (pos[0] - mx)**2 + (pos[1] - my)**2)
            
            for tx, ty in candidates:
                placed[m] = (tx, ty)
                if search(idx + 1):
                    return True
                    
        return False
        
    if search(0):
        return placed
    else:
        print("  [Macro Legalizer] DFS backtracking search timed out or failed. Falling back to greedy placement.")
        # Greedy fallback
        placed = {}
        sorted_macros = sorted(macros, key=lambda m: (-sizes[m][0]*sizes[m][1], positions[m][0], positions[m][1]))
        for m in sorted_macros:
            mx, my = positions[m]
            mw, mh = sizes[m]
            mx_clamped = int(np.clip(np.round(mx), llx, urx - mw))
            my_clamped = int(np.clip(np.round(my), lly, ury - mh))
            
            def overlaps_any_greedy(x, y):
                for pm, (px, py) in placed.items():
                    pw, ph = sizes[pm]
                    if max(x, px) < min(x + mw, px + pw) and max(y, py) < min(y + mh, py + ph):
                        return True
                return False
                
            if not overlaps_any_greedy(mx_clamped, my_clamped):
                placed[m] = (mx_clamped, my_clamped)
                continue
                
            found = False
            best_x, best_y = mx_clamped, my_clamped
            for s in range(1, max_steps):
                for ds in range(-s, s + 1):
                    for dy in [-s, s]:
                        tx = int(np.clip(mx_clamped + ds * step_x, llx, urx - mw))
                        ty = int(np.clip(my_clamped + dy * step_y, lly, ury - mh))
                        if not overlaps_any_greedy(tx, ty):
                            best_x, best_y = tx, ty
                            found = True
                            break
                    if found: break
                if found: break
                for dy in range(-s + 1, s):
                    for ds in [-s, s]:
                        tx = int(np.clip(mx_clamped + ds * step_x, llx, urx - mw))
                        ty = int(np.clip(my_clamped + dy * step_y, lly, ury - mh))
                        if not overlaps_any_greedy(tx, ty):
                            best_x, best_y = tx, ty
                            found = True
                            break
                    if found: break
                if found: break
            placed[m] = (best_x, best_y)
        return placed


def legalize_cells_on_rows(data, assignment, cell_positions):
    """
    Snaps standard cells to rows and resolves overlaps.
    Resolves macro overlaps first, then standard cells avoid macros.
    Standard cells will search nearby rows if the closest row is full.
    """
    legalized_positions = {}
    llx, lly, urx, ury = data.die_size
    
    # Process top and bottom dies separately
    for die in ['top', 'bottom']:
        die_insts = [name for name, d in assignment.items() if d == die]
        if not die_insts:
            continue
            
        tech_name = data.top_die_tech if die == 'top' else data.bottom_die_tech
        tech = data.technologies[tech_name]
        
        # Split into macros and standard cells
        macros = []
        std_cells = []
        sizes = {}
        for name in die_insts:
            inst = data.instances[name]
            lc = tech.lib_cells[inst.lib_cell_name]
            sizes[name] = (lc.size_x, lc.size_y)
            if lc.is_macro == 'Y':
                macros.append(name)
            else:
                std_cells.append(name)
                
        # 1. Legalize Macros (remove overlaps)
        legalized_macros = resolve_macro_overlaps(macros, sizes, cell_positions, data.die_size)
        legalized_positions.update(legalized_macros)
            
        # 2. Legalize Standard Cells to Rows
        rows = data.top_die_rows if die == 'top' else data.bottom_die_rows
        # A list of row Y coordinates and their heights
        row_ys = []
        row_heights = {}
        for r in rows:
            for r_idx in range(r.repeat_count):
                ry = r.start_y + r_idx * r.row_height
                row_ys.append(ry)
                row_heights[ry] = r.row_height
        row_ys = sorted(list(set(row_ys)))
        
        # Build macro blocked intervals for each row Y
        row_blocks = {ry: [] for ry in row_ys}
        for m_name, (mx, my) in legalized_macros.items():
            mw, mh = sizes[m_name]
            for ry in row_ys:
                r_h = row_heights[ry]
                if max(ry, my) < min(ry + r_h, my + mh):
                    row_blocks[ry].append((mx, mx + mw))
                    
        # Sort blocks
        for ry in row_ys:
            row_blocks[ry] = sorted(row_blocks[ry])

            
        # Sort standard cells by their current X position
        std_cells.sort(key=lambda name: cell_positions[name][0])
        
        # Greedy row assignment while avoiding macros and other standard cells
        placed_std_on_row = {ry: [] for ry in row_ys}
        
        for sc_name in std_cells:
            inst = data.instances[sc_name]
            lc = tech.lib_cells[inst.lib_cell_name]
            cell_w = lc.size_x
            cx, cy = cell_positions[sc_name]
            
            # Sort rows by distance to standard cell Y position
            sorted_rows = sorted(row_ys, key=lambda ry: abs(cy - ry))
            
            best_tx = None
            best_ry = None
            
            for ry in sorted_rows:
                # Build blocked intervals on this row
                blocks = []
                blocks.extend(row_blocks[ry])
                for placed_sc in placed_std_on_row[ry]:
                    sc_x = legalized_positions[placed_sc][0]
                    sc_w = sizes[placed_sc][0]
                    blocks.append((sc_x, sc_x + sc_w))
                    
                # Merge blocks
                blocks = sorted(blocks)
                merged_blocks = []
                for start, end in blocks:
                    if not merged_blocks or merged_blocks[-1][1] < start:
                        merged_blocks.append((start, end))
                    else:
                        merged_blocks[-1] = (merged_blocks[-1][0], max(merged_blocks[-1][1], end))
                        
                # Check closest tx
                tx_init = int(np.clip(np.round(cx), llx, urx - cell_w))
                
                def is_valid_pos(x):
                    for start, end in merged_blocks:
                        if max(x, start) < min(x + cell_w, end):
                            return False
                    return True
                    
                if is_valid_pos(tx_init):
                    best_tx = tx_init
                    best_ry = ry
                    break
                else:
                    # Search outwards
                    found_on_row = False
                    max_offset = urx - llx
                    for offset in range(1, max_offset):
                        # Check right
                        r_tx = tx_init + offset
                        if r_tx <= urx - cell_w and is_valid_pos(r_tx):
                            best_tx = r_tx
                            best_ry = ry
                            found_on_row = True
                            break
                        # Check left
                        l_tx = tx_init - offset
                        if l_tx >= llx and is_valid_pos(l_tx):
                            best_tx = l_tx
                            best_ry = ry
                            found_on_row = True
                            break
                    if found_on_row:
                        break
                        
            # Fallback if it does not fit anywhere without overlap
            if best_tx is None:
                best_ry = sorted_rows[0]
                best_tx = int(np.clip(np.round(cx), llx, urx - cell_w))
                
            legalized_positions[sc_name] = (best_tx, int(best_ry))
            placed_std_on_row[best_ry].append(sc_name)
            
    return legalized_positions

if __name__ == "__main__":
    from parser import parse_input
    from partitioner import spectral_partition
    from nesterov_placer import run_nesterov_placer
    
    test_file = "toy_example.txt"
    parsed = parse_input(test_file)
    partition = {
        'C1': 'top',
        'C2': 'bottom', # force some crossing nets
        'C3': 'top',
        'C4': 'bottom'
    }
    
    positions = {
        'C1': (100.0, 100.0),
        'C2': (120.0, 110.0),
        'C3': (200.0, 200.0),
        'C4': (220.0, 210.0)
    }
    
    print("Testing Terminal Legalization...")
    terms = optimize_and_legalize_terminals(parsed, partition, positions)
    print("Legalized Terminals:")
    for net, pos in terms.items():
        print(f"  {net}: {pos}")
        
    print("\nTesting Row Legalization...")
    leg_cells = legalize_cells_on_rows(parsed, partition, positions)
    print("Legalized Cells:")
    for cell, pos in leg_cells.items():
        print(f"  {cell}: {pos}")
