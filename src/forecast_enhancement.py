import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def forecast_enhanced(input_path="data/sales_cleaned.csv", output_dir="forecast_output", forecast_days=14):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load cleaned sales data
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Forecast for each product
    products = df["Product"].unique()
    for product in products:
        print(f"🔮 Forecasting for: {product}")
        product_df = df[df["Product"] == product]

        # Aggregate daily sales
        daily_sales = product_df.groupby("Date")["Qty_Sold"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]

        # Initialize and fit model
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(daily_sales)

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # Save forecast to CSV
        forecast_path = os.path.join(output_dir, f"{product}_forecast.csv")
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(forecast_path, index=False)

        # Plot forecast
        fig = model.plot(forecast)
        plt.title(f"{product} - {forecast_days}-Day Forecast")
        plt.xlabel("Date")
        plt.ylabel("Predicted Sales")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{product}_forecast.png"))
        plt.close()

    print(f"✅ Forecasts saved in '{output_dir}' for all products.")

# Run the function
if __name__ == "__main__":
    forecast_enhanced()