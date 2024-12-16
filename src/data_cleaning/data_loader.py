import pandas as pd

def load_data(file_path):
    """
    Loads tweet data from a specified file and organizes it into a pandas DataFrame.
    
    Each line in the input file is expected to contain an ID and the corresponding tweet text,
    separated by the first comma. The function processes each line, extracts the ID and text,
    and constructs a DataFrame with 'ID' as the index and 'Text' as the column.
    
    Args:
        file_path (str): The path to the input file containing tweet data.
    
    Returns:
        pd.DataFrame: A DataFrame with tweet IDs as the index and tweet texts as the data.
    """
    # Initialize a list to store the extracted ID and Text pairs
    rows = []
    
    # Open and read the input file with UTF-8 encoding
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Remove any leading/trailing whitespace and split the line at the first comma
            split_line = line.strip().split(",", 1)
            
            # Check if the line contains both ID and Text
            if len(split_line) == 2:
                rows.append(split_line)
    
    # Create a DataFrame from the list of [ID, Text] pairs
    df = pd.DataFrame(rows, columns=["ID", "Text"]).set_index("ID")
    
    return df
