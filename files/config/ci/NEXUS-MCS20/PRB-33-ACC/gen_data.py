import os
import json

# === Input config directory ===
input_dir = "../../NEXUS-MCS10/PRB-33-ACC/"  # ← Replace this with your input folder path

# === Output directory ===
output_dir = os.path.dirname(os.path.abspath(__file__))  # Where the script is located

# === Process each file ===
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as f:
            data = json.load(f)

        # Modify MCS indices
        if "ul_mcs" in data:
            data["ul_mcs"]["mcs_index"] = 20
        if "dl_mcs" in data:
            data["dl_mcs"]["mcs_index"] = 20

        # Save to new file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

print("All config files processed and saved to script directory.")
