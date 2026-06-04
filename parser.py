import os

class LibCell:
    def __init__(self, is_macro, name, size_x, size_y, pin_count):
        self.is_macro = is_macro  # 'Y' for Macro, 'N' for StdCell
        self.name = name
        self.size_x = size_x
        self.size_y = size_y
        self.pin_count = pin_count
        self.pins = {}  # pinName -> (loc_x, loc_y)

class Technology:
    def __init__(self, name):
        self.name = name
        self.lib_cells = {}  # libCellName -> LibCell

class Instance:
    def __init__(self, name, lib_cell_name):
        self.name = name
        self.lib_cell_name = lib_cell_name

class Net:
    def __init__(self, name, pin_count):
        self.name = name
        self.pin_count = pin_count
        self.pins = []  # List of (inst_name, pin_name)

class DieRow:
    def __init__(self, start_x, start_y, row_length, row_height, repeat_count):
        self.start_x = start_x
        self.start_y = start_y
        self.row_length = row_length
        self.row_height = row_height
        self.repeat_count = repeat_count

class PlacementInput:
    def __init__(self):
        self.technologies = {}  # techName -> Technology
        self.die_size = (0, 0, 0, 0)  # llx, lly, urx, ury
        self.top_die_max_util = 0.0
        self.bottom_die_max_util = 0.0
        self.top_die_rows = []
        self.bottom_die_rows = []
        self.top_die_tech = ""
        self.bottom_die_tech = ""
        self.terminal_size = (0, 0)
        self.terminal_spacing = 0
        self.terminal_cost = 0
        self.instances = {}  # instName -> Instance
        self.nets = {}  # netName -> Net

def parse_input(filename):
    data = PlacementInput()
    
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
        
        if keyword == "NumTechnologies":
            num_techs = int(tokens[1])
            idx += 1
            for _ in range(num_techs):
                while idx < num_lines and lines[idx][0] != "Tech":
                    idx += 1
                tech_tokens = lines[idx]
                tech_name = tech_tokens[1]
                num_lib_cells = int(tech_tokens[2])
                tech = Technology(tech_name)
                idx += 1
                
                for _ in range(num_lib_cells):
                    while idx < num_lines and lines[idx][0] != "LibCell":
                        idx += 1
                    lc_tokens = lines[idx]
                    is_macro = lc_tokens[1]  # 'Y' or 'N'
                    lc_name = lc_tokens[2]
                    size_x = int(lc_tokens[3])
                    size_y = int(lc_tokens[4])
                    pin_cnt = int(lc_tokens[5])
                    
                    lib_cell = LibCell(is_macro, lc_name, size_x, size_y, pin_cnt)
                    idx += 1
                    
                    for _ in range(pin_cnt):
                        while idx < num_lines and lines[idx][0] != "Pin":
                            idx += 1
                        pin_tokens = lines[idx]
                        pin_name = pin_tokens[1]
                        pin_x = int(pin_tokens[2])
                        pin_y = int(pin_tokens[3])
                        lib_cell.pins[pin_name] = (pin_x, pin_y)
                        idx += 1
                    
                    tech.lib_cells[lc_name] = lib_cell
                data.technologies[tech_name] = tech
                
        elif keyword == "DieSize":
            data.die_size = (int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4]))
            idx += 1
            
        elif keyword == "TopDieMaxUtil":
            data.top_die_max_util = float(tokens[1])
            idx += 1
            
        elif keyword == "BottomDieMaxUtil":
            data.bottom_die_max_util = float(tokens[1])
            idx += 1
            
        elif keyword == "TopDieRows":
            row = DieRow(int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5]))
            data.top_die_rows.append(row)
            idx += 1
            
        elif keyword == "BottomDieRows":
            row = DieRow(int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5]))
            data.bottom_die_rows.append(row)
            idx += 1
            
        elif keyword == "TopDieTech":
            data.top_die_tech = tokens[1]
            idx += 1
            
        elif keyword == "BottomDieTech":
            data.bottom_die_tech = tokens[1]
            idx += 1
            
        elif keyword == "TerminalSize":
            data.terminal_size = (int(tokens[1]), int(tokens[2]))
            idx += 1
            
        elif keyword == "TerminalSpacing":
            data.terminal_spacing = int(tokens[1])
            idx += 1
            
        elif keyword == "TerminalCost":
            data.terminal_cost = int(tokens[1])
            idx += 1
            
        elif keyword == "NumInstances":
            num_insts = int(tokens[1])
            idx += 1
            for _ in range(num_insts):
                while idx < num_lines and lines[idx][0] != "Inst":
                    idx += 1
                inst_tokens = lines[idx]
                inst_name = inst_tokens[1]
                lib_cell_name = inst_tokens[2]
                data.instances[inst_name] = Instance(inst_name, lib_cell_name)
                idx += 1
                
        elif keyword == "NumNets":
            num_nets = int(tokens[1])
            idx += 1
            for _ in range(num_nets):
                while idx < num_lines and lines[idx][0] != "Net":
                    idx += 1
                net_tokens = lines[idx]
                net_name = net_tokens[1]
                pin_cnt = int(net_tokens[2])
                net = Net(net_name, pin_cnt)
                idx += 1
                
                for _ in range(pin_cnt):
                    while idx < num_lines and lines[idx][0] != "Pin":
                        idx += 1
                    pin_tokens = lines[idx]
                    pin_path = pin_tokens[1]  # format: C1/P1
                    inst_n, pin_n = pin_path.split('/')
                    net.pins.append((inst_n, pin_n))
                    idx += 1
                data.nets[net_name] = net
        else:
            idx += 1
            
    return data

if __name__ == "__main__":
    # Small test
    test_file = "toy_example.txt"
    if os.path.exists(test_file):
        parsed = parse_input(test_file)
        print("Parsing successful!")
        print(f"Num Techs: {len(parsed.technologies)}")
        print(f"Die Size: {parsed.die_size}")
        print(f"Top Die Tech: {parsed.top_die_tech}, Bottom Die Tech: {parsed.bottom_die_tech}")
        print(f"Terminal Size: {parsed.terminal_size}, Spacing: {parsed.terminal_spacing}, Cost: {parsed.terminal_cost}")
        print(f"Num Instances: {len(parsed.instances)}")
        print(f"Num Nets: {len(parsed.nets)}")
        for net_name, net in parsed.nets.items():
            print(f"  Net {net_name} has pins: {net.pins}")
