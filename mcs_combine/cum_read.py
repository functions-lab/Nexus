import os
import subprocess
import csv

# Directory containing the CSV files
directory = "Agora-64-Half"

# MCS and zc mappings
mcs_mapping = [
    "MCS10", "MCS12", "MCS14", "MCS16", "MCS17", 
    "MCS18", "MCS20", "MCS22", "MCS24", "MCS26", "MCS28"
]
zc_mapping = [44, 56, 72, 88, 88, 88, 112, 128, 144, 176, 192]

# Collect data from each CSV file
data = []

for w in range(0, 11):
    for x in range(0, 11):
        for y in range(0, 11):
            for z in range(0, 11):
                file_name = f"{directory}/64-Task-Half-{w}-{x}-{y}-{z}.csv"
                try:
                    # Run the command and capture the output
                    command = f"python3 get_avg.py -f {file_name}"
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)

                    # Extract values from the command output
                    output = result.stdout
                    avg_time = float(output.split("Average Time:")[1].split("\n")[0].strip())
                    avg_retry_count = float(output.split("Average Retry Count:")[1].split("\n")[0].strip())
                    avg_bler = float(output.split("Average BLER:")[1].split("\n")[0].strip())

                    # Create the MCS pair string
                    mcs_pair = f"{mcs_mapping[w]}-{mcs_mapping[x]}-{mcs_mapping[y]}-{mcs_mapping[z]}"

                    # Calculate throughput
                    throughput = (16 * zc_mapping[w] * 22 + 16 * zc_mapping[x] * 22 + 16 * zc_mapping[y] * 22 + 16 * zc_mapping[z] * 22) / (avg_time / 1000) / 1000000000
                    throughput = round(throughput, 3)  # Rounding to 3 decimal places for clarity

                    data.append([mcs_pair, avg_time, avg_retry_count, avg_bler, throughput])
                except FileNotFoundError:
                    print(f"File {file_name} not found. Skipping.")
                except (IndexError, ValueError):
                    print(f"Error parsing output for file {file_name}. Skipping.")

# Write the collected data to a new CSV file
output_file = "64-Task-TP-aggregated_results.csv"
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["MCS Pair", "Average Time", "Average Retry Count", "Average BLER", "Throughput (Gbps)"])
    writer.writerows(data)

print(f"Data successfully written to {output_file}")
