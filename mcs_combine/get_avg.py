import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Calculate average of specified columns in a CSV file.")
parser.add_argument("-f", "--file", required=True, help="Path to the CSV file")

# Parse arguments
args = parser.parse_args()
file_path = args.file

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Calculate the averages for the specified columns
average_time = data['Time'].mean()
average_retry_count = data['Retry_Count'].mean()
average_BLER = data['BLER'].mean()

# Display the results
print("Average Time:", average_time)
print("Average Retry Count:", average_retry_count)
print("Average BLER:", average_BLER)

