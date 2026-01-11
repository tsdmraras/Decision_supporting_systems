import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_and_clean_data():
    """
    Loads the raw data, processes dates, and encodes categorical variables.
    """
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # 1. Load the dataset
    input_path = 'data/retail_sales_dataset.csv'
    if not os.path.exists(input_path):
        print(f"❌ Error: '{input_path}' not found. Please ensure the CSV file is in the 'data' folder.")
        return None

    df = pd.read_csv(input_path)
    
    # 2. Fix Date Format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 3. Feature Engineering: Extract Month, Year, and Day
    # These are crucial for Time Series Analysis later
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    
    # 4. Encode Categorical Variables (Text -> Numbers)
    # Models require numerical input. We use LabelEncoder for this.
    
    # Encode Gender (e.g., Male: 1, Female: 0)
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
    
    # Encode Product Category (e.g., Electronics: 1, Clothing: 0)
    le_category = LabelEncoder()
    df['Category_Encoded'] = le_category.fit_transform(df['Product Category'])
    
    # 5. Save the Processed Data
    output_path = 'data/cleaned_data.csv'
    df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"✅ Data processing complete.")
    print(f"📂 Processed file saved to: {output_path}")
    print("-" * 30)
    print("First 5 rows of processed data:")
    print(df[['Date', 'Gender', 'Gender_Encoded', 'Product Category', 'Category_Encoded']].head())
    
    return df

if __name__ == "__main__":
    load_and_clean_data()