import sys
import os
from parser import parse_input
from visualize_placement import parse_output

def calculate_die_utilization(data, placements, die_name):
    area_used = 0
    tech_name = data.top_die_tech if die_name == 'top' else data.bottom_die_tech
    tech = data.technologies.get(tech_name)
    if not tech: return 0

    for inst_name in placements[die_name]:
        inst = data.instances.get(inst_name)
        if inst and inst.lib_cell_name in tech.lib_cells:
            lib_cell = tech.lib_cells[inst.lib_cell_name]
            area_used += lib_cell.size_x * lib_cell.size_y
            
    die_width = data.die_size[2] - data.die_size[0]
    die_height = data.die_size[3] - data.die_size[1]
    total_area = die_width * die_height
    
    if total_area == 0:
        return 0
    return (area_used / total_area) * 100

def evaluate(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return None
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} not found.")
        return None

    data = parse_input(input_file)
    placements, terminals = parse_output(output_file)
    
    assignment = {}
    cell_positions = {}
    
    for inst_name, pos in placements['top'].items():
        assignment[inst_name] = 'top'
        cell_positions[inst_name] = pos
        
    for inst_name, pos in placements['bottom'].items():
        assignment[inst_name] = 'bottom'
        cell_positions[inst_name] = pos
        
    from main import calculate_hpwl_3d
    hpwl = calculate_hpwl_3d(data, assignment, cell_positions, terminals)
    
    num_terminals = len(terminals)
    terminal_cost = num_terminals * data.terminal_cost
    total_score = hpwl + terminal_cost
    
    top_util = calculate_die_utilization(data, placements, 'top')
    bottom_util = calculate_die_utilization(data, placements, 'bottom')
    
    return {
        "HPWL": hpwl,
        "NumTerminals": num_terminals,
        "TerminalCost": terminal_cost,
        "TotalScore": total_score,
        "TopUtil": top_util,
        "BottomUtil": bottom_util
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 evaluator.py <input_file> <output_file_1> [<output_file_2> ...]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_files = sys.argv[2:]
    
    results = {}
    for out_file in output_files:
        res = evaluate(input_file, out_file)
        if res:
            results[out_file] = res
            
    print("\n### 評估結果對比表")
    print()
    print("| 指標 (Metrics) | " + " | ".join([os.path.basename(f) for f in output_files]) + " |")
    print("|---| " + " | ".join(["---" for _ in output_files]) + " |")
    
    metrics = [
        ("Total Score", "TotalScore", "{:.2f}"),
        ("Total HPWL", "HPWL", "{:.2f}"),
        ("Number of Terminals", "NumTerminals", "{}"),
        ("Terminal Cost", "TerminalCost", "{:.2f}"),
        ("Top Die Util (%)", "TopUtil", "{:.2f}%"),
        ("Bottom Die Util (%)", "BottomUtil", "{:.2f}%")
    ]
    
    for label, key, fmt in metrics:
        row = f"| **{label}** |"
        for out_file in output_files:
            if out_file in results:
                val = results[out_file][key]
                row += f" {fmt.format(val)} |"
            else:
                row += " N/A |"
        print(row)
    print()

if __name__ == "__main__":
    main()
