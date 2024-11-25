import pandas as pd

def load_data(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
                # Split on the first comma
            split_line = line.strip().split(",", 1)
            if len(split_line) == 2:  # Ensure the row has both ID and Text
                rows.append(split_line)

        # Create a DataFrame from the processed rows
    df = pd.DataFrame(rows, columns=["ID", "Text"]).set_index("ID")
    return df