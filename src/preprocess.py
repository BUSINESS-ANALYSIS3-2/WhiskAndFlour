import pandas as pd

def preprocess_sales(input_path="data/sales.csv", output_path="data/sales_cleaned.csv"):
    # Load data
    df = pd.read_csv(input_path)

    # Convert 'Date' to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    # Sort by date
    df = df.sort_values("Date")

    # Remove duplicates (if any)
    df = df.drop_duplicates()

    # Check for missing values
    missing = df.isnull().sum()
    print("🔍 Missing values:\n", missing)

    # Fill or drop missing values (if needed)
    df = df.dropna()

    # Add weekday column
    df["Weekday"] = df["Date"].dt.day_name()

    # Add rolling average of Qty_Sold per product (7-day window)
    df["Rolling_Avg_Qty"] = (
        df.groupby("Product")["Qty_Sold"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned sales data saved to {output_path}")

# Run the function
preprocess_sales()
