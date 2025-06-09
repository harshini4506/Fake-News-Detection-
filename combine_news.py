import pandas as pd
import os

def load_and_prepare_dataset(file_path):
    """Load and prepare a dataset based on its format."""
    try:
        print(f"Attempting to read {file_path}...")
        # Explicitly specify dtype for the 'Label' column to prevent misinterpretation
        df = pd.read_csv(file_path, dtype={'Label': str})
        print(f"Successfully read {file_path}.")
        print(f"Columns in {os.path.basename(file_path)}: {df.columns.tolist()}")
        
        # Assuming the format is the new format (Statement, Label)
        df = df.rename(columns={'Statement': 'text'})
        
        # Debugging: print original Label column info after reading
        if 'Label' in df.columns:
            print(f"Original Label column dtype in {os.path.basename(file_path)}: {df['Label'].dtype}")
            print(f"First 5 values of original Label column in {os.path.basename(file_path)}: {df['Label'].head().tolist()}")
            
            # Map TRUE/FALSE to REAL/FAKE
            df['label'] = df['Label'].map({'TRUE': 'REAL', 'FALSE': 'FAKE'})
        else:
             print(f"Warning: 'Label' column not found in {file_path}. Cannot process labels.")
             return None
        
        # Print unique labels after mapping
        if 'label' in df.columns:
             print(f"Unique labels in {os.path.basename(file_path)} after mapping: {df['label'].unique()}")

        # Select final columns and drop rows where 'text' or 'label' is NaN
        df = df[['text', 'label']].dropna(subset=['text', 'label'])
        
        # Add source information
        df['source'] = os.path.basename(file_path).replace('.csv', '')

        return df
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load new datasets
    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    # valid_path = os.path.join('data', 'valid.csv') # Exclude valid.csv for now
    
    # Initialize empty list to store all dataframes
    all_dfs = []
    
    # Load train and test datasets if they exist
    for path in [train_path, test_path]: # Only process train and test
        if os.path.exists(path):
            print(f"\nLoading {path}...")
            df = load_and_prepare_dataset(path)
            if df is not None and not df.empty:
                all_dfs.append(df)
                print(f"Loaded {len(df)} valid articles from {path}")
                print(f"  - REAL articles: {len(df[df['label'] == 'REAL'])}")
                print(f"  - FAKE articles: {len(df[df['label'] == 'FAKE'])}")
            elif df is not None and df.empty:
                 print(f"No valid articles loaded from {path} after cleaning.")
    
    if not all_dfs:
        print("No datasets found. Please make sure at least one dataset is available.")
        return
    
    # Combine all datasets
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print("\nBefore final duplicate removal and saving:")
    print(f"Total articles: {len(combined)}")
    print(f"REAL articles: {len(combined[combined['label'] == 'REAL'])}")
    print(f"FAKE articles: {len(combined[combined['label'] == 'FAKE'])}")
    
    # Show duplicates by source
    print("\nDuplicate analysis by source:")
    for source in combined['source'].unique():
        source_df = combined[combined['source'] == source]
        duplicates = source_df[source_df.duplicated(subset=['text'], keep=False)]
        print(f"{source}: {len(duplicates)} duplicates out of {len(source_df)} articles")
    
    # Remove duplicates based on text, keeping the first occurrence
    combined = combined.drop_duplicates(subset=['text'], keep='first')
    
    # Shuffle the combined dataset
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the combined dataset
    output_path = os.path.join('data', 'news.csv')
    combined.to_csv(output_path, index=False)
    
    print("\nAfter final duplicate removal and saving:")
    print(f"Total articles: {len(combined)}")
    print(f"REAL articles: {len(combined[combined['label'] == 'REAL'])}")
    print(f"FAKE articles: {len(combined[combined['label'] == 'FAKE'])}")
    print(f"\nCombined dataset saved to {output_path}")

if __name__ == '__main__':
    main() 