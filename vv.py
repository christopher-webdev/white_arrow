# import pandas as pd

# # === File paths ===
# original_file = "gbpusd.csv"
# forward_test_file = "gbpusdcut.csv"

# # === Parameters ===
# rows_to_extract = 20000

# # === Step 1: Load the original CSV ===
# df = pd.read_csv(original_file)#,sep="\t",encoding="utf-16")

# # === Step 2: Split the last N rows ===
# df_forward = df.tail(rows_to_extract)
# df_remaining = df.iloc[:-rows_to_extract]

# # === Step 3: Save the extracted rows to a new CSV ===
# df_forward.to_csv(forward_test_file, index=False)

# # === Step 4: Overwrite original CSV with remaining rows ===
# df_remaining.to_csv(original_file, index=False)

# print(f"✅ Extracted last {rows_to_extract} rows to '{forward_test_file}'")
# print(f"✅ Original file '{original_file}' now contains {len(df_remaining)} rows.")



















import pandas as pd
import numpy as np

# Define the file paths
# input_file = "5m-1.csv"
input_file = "xauusd.csv"
# df = pd.read_csv("XAUUSD.csv", sep="\t", )
df = pd.read_csv(input_file, sep="\t",encoding="utf-16" )     # Change to your actual file name
output_file = "xauusd.csv"  # New file with reversed order

# Load the CSV file with UTF-16 encoding and tab delimiter
# df = pd.read_csv(input_file)

# Reverse the DataFrame order (newest to oldest → oldest to newest)
df = df.iloc[::-1].reset_index(drop=True)
# df.sort_values("Time").reset_index(drop=True)
# df = df.iloc[8020:].reset_index(drop=True)
# Save the reversed file in UTF-16 with tab delimiter
df.to_csv(output_file,index=False )
#sep="\t", encoding="utf-16", 
# print("✅ File reversed successfully and saved as:", output_file)


print(df.head())  # Should now show the earliest time first
# # Try converting Volume to numeric to detect problematic rows
# df['Volume_numeric'] = pd.to_numeric(df['Volume'], errors='coerce')

# # Find rows where conversion failed (i.e., invalid Volume entries)
# invalid_volume_rows = df[df['Volume_numeric'].isna() & df['Volume'].notna()]

# # Show the index and original Volume values
# print("🔍 Rows with non-numeric Volume values:\n")
# print(invalid_volume_rows[['Volume']])
