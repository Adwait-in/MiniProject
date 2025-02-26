import pandas as pd
import os

# Get the current directory
folder_path = os.getcwd()

# Get all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Read and combine all CSV files
combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('hand_gesture_data_combined.csv', index=False)

print(f"Combined {len(csv_files)} CSV files and saved as 'combined_file.csv'")