import csv

def format_csv_features_to_file(csv_file_path, output_file):
    # Read the header from the CSV file
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Get the first row (header)
    
    # Count the total number of features
    feature_count = len(header)
    print(f"Total number of features: {feature_count}")
    
    # Format features with double quotes, 10 per row
    formatted_lines = []
    for i in range(0, feature_count, 10):
        batch = header[i:i+10]
        quoted_features = [f'"{feature}"' for feature in batch]
        formatted_lines.append(", ".join(quoted_features))
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write(",\n".join(formatted_lines))
    
    print(f"Formatted features saved to {output_file}")

# Example usage:
format_csv_features_to_file('classified_regression_data_sell2.csv', 'formatted_features.txt')