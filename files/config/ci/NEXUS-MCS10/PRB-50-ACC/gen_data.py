import os

# === Input directory with the config files ===
input_dir = "../../NEXUS-MCS17/PRB-50-ACC/"  # ← Replace this with your input folder path
output_dir = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        # Replace any line that sets mcs_index to 17
        new_lines = []
        for line in lines:
            if '"mcs_index": 17' in line:
                line = line.replace('"mcs_index": 17', '"mcs_index": 10')
            new_lines.append(line)

        with open(output_path, 'w') as f:
            f.writelines(new_lines)

print("All mcs_index values set to 10 and files saved in script directory.")
