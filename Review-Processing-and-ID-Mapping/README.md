# Review Processing and ID Mapping

This project provides Python scripts to process review datasets by:
- Generating unique `User ID`s for each user based on their name.
- Extracting important words (proper nouns and adjectives) from the `Review Text` field.
- Assigning unique `Tag ID`s to important words, ensuring that identical words get the same `Tag ID`.
- Normalizing the words by removing extra spaces and making them case-insensitive.
- Saving the cleaned and structured data in a CSV format.

### Features:
- Adds a `User ID` column based on unique users.
- Extracts key words (proper nouns and adjectives) and assigns unique IDs to them.
- Ensures that identical users and tags share the same ID.
- The dataset is normalized and saved as a CSV file for easy integration into other projects.

### How to Use:
1. Set the path to your dataset in the `file_path` variable.
2. Run the Python script to process the dataset.
3. The processed data will be saved as `processed_dataset.csv`.

### Requirements:
- pandas
- nltk
