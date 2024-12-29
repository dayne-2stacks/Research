import os
import csv


if __name__ == "__main__":
    # File paths
    training_paths = ["R1", "R2", "R3"]
    testing_paths = ["R4"]
    validation_paths = ["R5"]

    fileDict = {
        "training.csv": training_paths,
        "testing.csv": testing_paths,
        "validation.csv": validation_paths
    }

    # Category mapping
    category_mapping = {
        "plain_arch": 0,
        "right_loop": 1,
        "left_loop": 2,
        "tented_arch": 3,
        "whorl": 4,
    }

    # Access file names and paths for each set
    for file_name, paths in fileDict.items():
        with open(file_name, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["file", "categoryNumber"])  # Write header

            # Access each path provided
            for path in paths:
                for filename in os.listdir(path):
                    if filename.endswith('.tsv'):  # Only process .tsv files
                        # Extract the number and category from the filename
                        num_category = filename.split('_', 1)
                        num = num_category[0]
                        category = num_category[1].replace('.tsv', '')

                        # Map the category to a number
                        category_number = category_mapping.get(category, -1)  # -1 if category not found

                        # Write the file path and category number to the CSV file
                        writer.writerow([f'{path}/{filename}', category_number])

        print(f"CSV file '{file_name}' has been created successfully.")
