import os
import re

# === Input directory ===
input_dir = "../../NEXUS-MCS10/PRB-50-ACC/"

# === Revert mcs_index: 12 to mcs_index: 10 ===
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(input_dir, filename)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        modified = False
        new_lines = []
        for line in lines:
            new_line = re.sub(r'("mcs_index"\s*:\s*)12', r'\g<1>10', line)
            if new_line != line:
                modified = True
            new_lines.append(new_line)

        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)

print("Reverted all mcs_index values from 12 to 10.")
