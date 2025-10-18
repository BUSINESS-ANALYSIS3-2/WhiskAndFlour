import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

def generate_sales(
    start_date="2024-01-01",
    end_date="2024-08-31",
    output_path="data/sales.csv"
):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define products and prices
    products = {
        "Croissant": 15,
        "Baguette": 20,
        "Vegan Chocolate Tart": 25,
        "Cinnamon Roll": 18,
        "Birthday Cake": 150
    }

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Generate sales data
    rows = []
    for date in date_range:
        weekday = date.weekday()  # 0 = Monday, 6 = Sunday
        for product, price in products.items():
            # Simulate higher weekend demand
            qty = random.randint(10, 35) if weekday in [5, 6] else random.randint(5, 20)
            revenue = qty * price
            rows.append([date.strftime("%Y-%m-%d"), product, qty, price, revenue])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Date", "Product", "Qty_Sold", "Unit_Price", "Total_Revenue"])

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Sales data generated and saved to {output_path}")

# Run the function
generate_sales()
