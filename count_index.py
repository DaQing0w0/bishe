import pandas as pd

try:
    # Load the CSV file
    df = pd.read_csv('epoch2_processed.csv')
    
    # Count occurrences of 'calculated_index'
    counts = df['calculated_index'].value_counts().sort_index()
    
    print("Top 10 most frequent calculated_index:")
    print(df['calculated_index'].value_counts().head(10))
    
    output_file = 'epoch2_index_counts.csv'
    counts.to_csv(output_file, header=['count'])
    print(f"\nFull counts saved to {output_file}")
except Exception as e:
    print(f"Error: {e}")
