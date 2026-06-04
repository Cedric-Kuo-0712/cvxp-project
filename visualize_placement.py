import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from parser import parse_input

def parse_output(filename):
    """
    Parses the placement output file.
    """
    placements = {'top': {}, 'bottom': {}}
    terminals = {}
    
    if not os.path.exists(filename):
        print(f"Error: Output file {filename} not found.")
        return placements, terminals
        
    with open(filename, 'r') as f:
        lines = [line.strip().split() for line in f if line.strip()]
        
    idx = 0
    num_lines = len(lines)
    
    while idx < num_lines:
        tokens = lines[idx]
        if not tokens:
            idx += 1
            continue
        keyword = tokens[0]
        
        if keyword == "TopDiePlacement":
            num_insts = int(tokens[1])
            idx += 1
            for _ in range(num_insts):
                if idx >= num_lines:
                    break
                inst_tokens = lines[idx]
                inst_name = inst_tokens[1]
                x = int(inst_tokens[2])
                y = int(inst_tokens[3])
                placements['top'][inst_name] = (x, y)
                idx += 1
                
        elif keyword == "BottomDiePlacement":
            num_insts = int(tokens[1])
            idx += 1
            for _ in range(num_insts):
                if idx >= num_lines:
                    break
                inst_tokens = lines[idx]
                inst_name = inst_tokens[1]
                x = int(inst_tokens[2])
                y = int(inst_tokens[3])
                placements['bottom'][inst_name] = (x, y)
                idx += 1
                
        elif keyword == "NumTerminals":
            num_terms = int(tokens[1])
            idx += 1
            for _ in range(num_terms):
                if idx >= num_lines:
                    break
                term_tokens = lines[idx]
                net_name = term_tokens[1]
                x = int(term_tokens[2])
                y = int(term_tokens[3])
                terminals[net_name] = (x, y)
                idx += 1
        else:
            idx += 1
            
    return placements, terminals

def get_instance_info(data, inst_name, die):
    """
    Get size and type of the instance.
    """
    tech_name = data.top_die_tech if die == 'top' else data.bottom_die_tech
    tech = data.technologies[tech_name]
    inst = data.instances[inst_name]
    lc = tech.lib_cells[inst.lib_cell_name]
    return lc.size_x, lc.size_y, lc.is_macro

def visualize(input_file, output_file, image_path):
    print(f"Loading files:\n  Input: {input_file}\n  Output: {output_file}")
    data = parse_input(input_file)
    placements, terminals = parse_output(output_file)
    
    # Die boundary
    llx, lly, urx, ury = data.die_size
    die_w = urx - llx
    die_h = ury - lly
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor='#F8F9FA')
    
    # Colors and styles
    top_macro_color = '#FF8787'      # Light coral
    top_macro_edge = '#C92A2A'
    top_std_color = '#74C0FC'        # Sky blue
    top_std_edge = '#1C7ED6'
    
    bot_macro_color = '#FFD43B'      # Mustard yellow
    bot_macro_edge = '#F59F00'
    bot_std_color = '#63E6BE'        # Mint green
    bot_std_edge = '#0CA678'
    
    terminal_color = '#BE4BDB'       # Grape purple
    terminal_edge = '#862E9C'
    
    net_color = '#868E96'            # Gray
    
    # Reconstruct assignment from placement
    assignment = {}
    cell_positions = {}
    for die in ['top', 'bottom']:
        for inst_name, pos in placements[die].items():
            assignment[inst_name] = die
            cell_positions[inst_name] = pos
            
    # Subplot details
    dies = ['top', 'bottom']
    titles = ['Top Die Placement', 'Bottom Die Placement']
    
    for i, die in enumerate(dies):
        ax = axs[i]
        ax.set_facecolor('#FFFFFF')
        ax.set_title(titles[i], fontsize=14, fontweight='bold', pad=15)
        
        # 1. Draw Die Boundary
        rect_die = patches.Rectangle((llx, lly), die_w, die_h, linewidth=2.5, edgecolor='#343A40', facecolor='none', zorder=5)
        ax.add_patch(rect_die)
        
        # 2. Draw Die Rows (subtle background grid)
        rows = data.top_die_rows if die == 'top' else data.bottom_die_rows
        for row in rows:
            for r_idx in range(row.repeat_count):
                ry = row.start_y + r_idx * row.row_height
                ax.axhline(y=ry, color='#E9ECEF', linestyle=':', linewidth=0.8, zorder=1)
                
        # 3. Draw Instances
        die_placements = placements[die]
        num_insts = len(die_placements)
        
        for inst_name, (cx, cy) in die_placements.items():
            w, h, is_macro = get_instance_info(data, inst_name, die)
            
            if is_macro == 'Y':
                face = top_macro_color if die == 'top' else bot_macro_color
                edge = top_macro_edge if die == 'top' else bot_macro_edge
                label_text = f"{inst_name}\n(Macro)"
                z = 4
            else:
                face = top_std_color if die == 'top' else bot_std_color
                edge = top_std_edge if die == 'top' else bot_std_edge
                label_text = inst_name
                z = 3
                
            rect = patches.Rectangle((cx, cy), w, h, linewidth=1.2, edgecolor=edge, facecolor=face, alpha=0.75, zorder=z)
            ax.add_patch(rect)
            
            # Label
            if num_insts <= 100:
                ax.text(cx + w/2.0, cy + h/2.0, label_text, color='#212529', fontsize=8,
                        ha='center', va='center', weight='bold', zorder=z+1)
                
        # 4. Draw Terminals
        tw, th = data.terminal_size
        for net_name, (tx, ty) in terminals.items():
            term_rect = patches.Rectangle((tx - tw/2.0, ty - th/2.0), tw, th, linewidth=1.5,
                                          edgecolor=terminal_edge, facecolor=terminal_color, alpha=0.9, zorder=6)
            ax.add_patch(term_rect)
            
            if len(terminals) <= 30:
                ax.text(tx, ty + th/2.0 + 1, f"T:{net_name}", color='#862E9C', fontsize=7,
                        ha='center', va='bottom', weight='bold', zorder=7)
                
        # 5. Draw Nets (Connections)
        # To avoid clutter, only draw nets if requested or case is small
        draw_nets = True # Draw them cleanly
        if draw_nets:
            for net_name, net in data.nets.items():
                # Filter pins belonging to this die
                die_pins = []
                has_terminal = net_name in terminals
                
                for inst_name, pin_name in net.pins:
                    if inst_name in die_placements:
                        cx, cy = die_placements[inst_name]
                        # get pin offset
                        inst = data.instances[inst_name]
                        tech_name = data.top_die_tech if die == 'top' else data.bottom_die_tech
                        tech = data.technologies[tech_name]
                        lc = tech.lib_cells[inst.lib_cell_name]
                        px_off, py_off = lc.pins.get(pin_name, (0, 0))
                        die_pins.append((cx + px_off, cy + py_off))
                        
                if not die_pins:
                    continue
                    
                if has_terminal:
                    # Draw connection lines from pins to terminal center
                    tx, ty = terminals[net_name]
                    for px, py in die_pins:
                        ax.plot([px, tx], [py, ty], color=net_color, linestyle='-', linewidth=0.8, alpha=0.4, zorder=2)
                else:
                    # Single-die net: draw lines from pins to net centroid
                    if len(die_pins) > 1:
                        mx = np.mean([p[0] for p in die_pins])
                        my = np.mean([p[1] for p in die_pins])
                        for px, py in die_pins:
                            ax.plot([px, mx], [py, my], color=net_color, linestyle='--', linewidth=0.7, alpha=0.3, zorder=2)
                            
        # Set limits with small margins
        margin = max(die_w, die_h) * 0.05
        ax.set_xlim(llx - margin, urx + margin)
        ax.set_ylim(lly - margin, ury + margin)
        ax.set_xlabel('X Coordinate', fontsize=11)
        ax.set_ylabel('Y Coordinate', fontsize=11)
        ax.grid(True, which='both', color='#E9ECEF', linestyle='-', linewidth=0.5)
        ax.set_aspect('equal')
        
    # Legend
    legend_elements = [
        patches.Patch(facecolor=top_std_color, edgecolor=top_std_edge, alpha=0.75, label='Top Std Cell'),
        patches.Patch(facecolor=top_macro_color, edgecolor=top_macro_edge, alpha=0.75, label='Top Macro'),
        patches.Patch(facecolor=bot_std_color, edgecolor=bot_std_edge, alpha=0.75, label='Bottom Std Cell'),
        patches.Patch(facecolor=bot_macro_color, edgecolor=bot_macro_edge, alpha=0.75, label='Bottom Macro'),
        patches.Patch(facecolor=terminal_color, edgecolor=terminal_edge, alpha=0.9, label='Terminal (TSV)'),
        plt.Line2D([0], [0], color=net_color, linestyle='-', linewidth=1, alpha=0.7, label='Cross-Die Net to Terminal'),
        plt.Line2D([0], [0], color=net_color, linestyle='--', linewidth=1, alpha=0.7, label='Intra-Die Net to Centroid')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    # Subtitle or details
    num_nets = len(data.nets)
    num_cross = len(terminals)
    total_insts = len(data.instances)
    
    fig.suptitle(f"3D IC Placement Results: {os.path.basename(input_file)}\n"
                 f"Instances: {total_insts} | Nets: {num_nets} | Terminals: {num_cross}",
                 fontsize=16, fontweight='bold', y=0.97)
                 
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig(image_path, dpi=150)
    plt.close()
    print(f"Visualization successfully saved to: {image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 visualize_placement.py <input_file> <output_file> [output_image_path]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if len(sys.argv) >= 4:
        image_path = sys.argv[3]
    else:
        # Default name based on the output file
        base = os.path.splitext(output_file)[0]
        image_path = f"{base}_visualization.png"
        
    visualize(input_file, output_file, image_path)
