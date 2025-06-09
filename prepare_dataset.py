import pandas as pd
import os

def main():
    data_path = os.path.join('data', 'news.csv')
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please make sure 'news.csv' is in the 'data' directory.")
        return
    # Example: just print the first few rows
    df = pd.read_csv(data_path)
    print(df.head())
    # Add any preprocessing steps here as needed
    # Save the processed dataset if required

if __name__ == '__main__':
    main() 