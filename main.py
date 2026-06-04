import sys
import os
import numpy as np
from parser import parse_input
from partitioner import spectral_partition
from nesterov_placer import run_nesterov_placer
from legalizer import legalize_cells_on_rows, optimize_and_legalize_terminals

def calculate_hpwl_3d(data, assignment, cell_positions, terminals):
    """
    Calculate the total HPWL of the 3D placement.
    Includes terminal center points for cross-die nets.
    """
    total_hpwl = 0.0
    
    for net_name, net in data.nets.items():
        # Get pin global coordinates
        pins_by_die = {'top': [], 'bottom': []}
        
        for inst_name, pin_name in net.pins:
            if inst_name not in assignment:
                continue
            die = assignment[inst_name]
            cx, cy = cell_positions[inst_name]
            
            tech_name = data.top_die_tech if die == 'top' else data.bottom_die_tech
            tech = data.technologies[tech_name]
            inst = data.instances[inst_name]
            lc = tech.lib_cells[inst.lib_cell_name]
            
            px_off, py_off = lc.pins.get(pin_name, (0, 0))
            pins_by_die[die].append((cx + px_off, cy + py_off))
            
        # Check if cross-die
        is_cross = len(pins_by_die['top']) > 0 and len(pins_by_die['bottom']) > 0
        
        if is_cross:
            # Must have a terminal
            tx, ty = terminals.get(net_name, (0, 0))
            
            # HPWL on top die
            xs_top = [p[0] for p in pins_by_die['top']] + [tx]
            ys_top = [p[1] for p in pins_by_die['top']] + [ty]
            hpwl_top = (max(xs_top) - min(xs_top)) + (max(ys_top) - min(ys_top))
            
            # HPWL on bottom die
            xs_bottom = [p[0] for p in pins_by_die['bottom']] + [tx]
            ys_bottom = [p[1] for p in pins_by_die['bottom']] + [ty]
            hpwl_bottom = (max(xs_bottom) - min(xs_bottom)) + (max(ys_bottom) - min(ys_bottom))
            
            total_hpwl += hpwl_top + hpwl_bottom
        else:
            # Single-die net
            die = 'top' if pins_by_die['top'] else 'bottom'
            pins = pins_by_die[die]
            if len(pins) <= 1:
                continue
            xs = [p[0] for p in pins]
            ys = [p[1] for p in pins]
            total_hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))
            
    return total_hpwl

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    print(f"=== Starting 3D Placement Engine ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # 1. Parse Input
    print("\n[Step 1] Parsing Input...")
    data = parse_input(input_file)
    print(f"  Parsed {len(data.instances)} instances, {len(data.nets)} nets.")
    
    # 2. Partition Netlist (Spectral Relaxation)
    print("\n[Step 2] Spectral Partitioning...")
    assignment = spectral_partition(data)
    
    # 3. Global Placement (Nesterov Accelerated Gradient & Electrostatics)
    print("\n[Step 3] Global Placement (NAG + Electrostatics)...")
    gp_positions = {}
    
    # Run placement for each die
    for die in ['top', 'bottom']:
        print(f"\n--- Running Global Placement for {die.upper()} Die ---")
        positions_die = run_nesterov_placer(data, assignment, die, num_iterations=50, lr=0.5)
        gp_positions.update(positions_die)
        
    # 4. Legalize cell positions (Snap to Rows & remove overlaps)
    print("\n[Step 4] Legalizing Cell Positions...")
    legal_positions = legalize_cells_on_rows(data, assignment, gp_positions)
    
    # 5. Optimize & Legalize Terminals (Median L1 + Grid Matching)
    print("\n[Step 5] Optimizing and Legalizing Terminals...")
    legal_terminals = optimize_and_legalize_terminals(data, assignment, legal_positions)
    print(f"  Placed and legalized {len(legal_terminals)} terminals.")
    
    # 6. Compute Final Metrics
    print("\n=== Final Quality Metrics ===")
    final_hpwl = calculate_hpwl_3d(data, assignment, legal_positions, legal_terminals)
    terminal_cost = len(legal_terminals) * data.terminal_cost
    total_score = final_hpwl + terminal_cost
    
    print(f"  Final HPWL: {final_hpwl:.2f}")
    print(f"  Number of Terminals: {len(legal_terminals)} (Cost per Terminal: {data.terminal_cost})")
    print(f"  Terminal Cost Component: {terminal_cost:.2f}")
    print(f"  Total Score (HPWL + Terminal Cost): {total_score:.2f}")
    
    # 7. Write Output file
    print(f"\n[Step 6] Writing Output file to {output_file}...")
    
    top_insts = [name for name, die in assignment.items() if die == 'top']
    bottom_insts = [name for name, die in assignment.items() if die == 'bottom']
    
    with open(output_file, 'w') as f:
        # Top Die
        f.write(f"TopDiePlacement {len(top_insts)}\n")
        for name in top_insts:
            x, y = legal_positions[name]
            f.write(f"Inst {name} {x} {y} R0\n")
            
        # Bottom Die
        f.write(f"BottomDiePlacement {len(bottom_insts)}\n")
        for name in bottom_insts:
            x, y = legal_positions[name]
            f.write(f"Inst {name} {x} {y} R0\n")
            
        # Terminals
        f.write(f"NumTerminals {len(legal_terminals)}\n")
        for name, (tx, ty) in legal_terminals.items():
            f.write(f"Terminal {name} {tx} {ty}\n")
            
    print("=== Placement Engine Finished Successfully! ===")
    
    # 8. Automatic Visualization
    print("\n[Step 7] Automatically generating placement visualization...")
    base = os.path.splitext(output_file)[0]
    image_path = f"{base}_visualization.png"
    try:
        from visualize_placement import visualize
        visualize(input_file, output_file, image_path)
    except Exception as e:
        print(f"  Warning: Automatic visualization failed: {e}")


if __name__ == "__main__":
    main()
