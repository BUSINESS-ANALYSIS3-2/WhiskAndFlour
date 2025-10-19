# Whisk & Flour — AI Demand Forecasting & Recommendation System

Whisk & Flour is an AI-powered solution that brings data-driven demand forecasting and a conversational recommendation chatbot to artisan bakeries. The system combines time-series forecasting to predict daily and weekly product demand with an NLP-based chatbot that understands customer dietary needs and suggests menu items and recipes. The result is reduced waste, improved stock availability, better customer experience, and increased revenue.

---

## Table of contents

- Project overview
- Business objectives & success criteria
- Key features
- Theoretical approach
- Data format & examples
- Quickstart (setup & run)
- Typical workflows
  - Demand forecasting
  - Chatbot & recommendation
- Monitoring, evaluation & metrics
- Architecture & components
- Risks & constraints
- Suggested roadmap
- File structure (recommended)
- License & contact

---

## Project overview

Whisk & Flour helps bakeries:
- Forecast demand per product (daily & weekly) using time-series models (e.g., Prophet).
- Generate inventory and production recommendations to reduce overproduction and stockouts.
- Provide a quick NLP chatbot that classifies customer intents (e.g., `vegan`, `gluten-free`) and suggests items or recipes.

This README documents the solution scope, data expectations, model approaches, quickstart steps, and operational considerations so development, deployment, and evaluation can proceed smoothly.

---

## Business objectives

1. Improve operational efficiency
   - Prevent over- and underproduction.
   - Reduce on-hand waste by optimizing purchasing and production.

2. Enhance customer experience
   - Personalized recipe and product recommendations via chatbot.
   - Faster ordering during peak times.

3. Increase profit
   - Align production with demand to reduce cost and increase sales.
   - Promote high-margin and trending products.

4. Leverage data for strategic decisions
   - Track sales trends to inform product innovation, promotions, and seasonal planning.

---

## Success criteria

- Forecasting accuracy: ≥ 85% day-to-day and week-to-week within first 3 months.
- Waste reduction: ≥ 30% reduction in waste from overproduction.
- Stock availability: 95% availability for best-selling items during peaks.
- Staff efficiency: 40% reduction in manual planning time.
- Chatbot adoption: 70% of customers use chatbot for orders/recommendations within 6 months.
- Customer satisfaction: ≥ 90% average rating for chatbot interactions.
- Wait-time reduction: ≥ 25% reduction during busy hours.
- Financial: 15% revenue increase within 1 year and positive ROI from savings and increased sales.

---

## Key features

- Time-series forecasting for each product (daily & weekly horizons).
- Inventory and production recommendations derived from forecasts.
- NLP conversational chatbot for dietary-aware recommendations (vegan, gluten-free, low-sugar, etc.).
- Dashboard for bakery staff showing forecasts, inventory suggestions, and chatbot analytics.
- Lightweight architecture suitable for small budgets and academic timelines (open-source libraries & free tiers).

---

## Theoretical approach

- Forecasting: Facebook Prophet (or equivalent) trained on historical daily sales per product. Models augmented with derived features (lags, rolling averages, weekday/month indicators).
- Recommendation/classification: Multinomial Naive Bayes (fast, effective for small training corpora) using tokenization + CountVectorizer or TF-IDF.
- Data processing: Pandas and NumPy for cleaning and feature engineering.
- Visualization: Matplotlib (or Plotly/Streamlit for interactive dashboards).

Example forecasting pseudo-code (illustrative):
```python
def forecast_enhanced(input_path="data/sales_cleaned.csv", output_dir="forecast_output", forecast_days=14):
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    products = df["Product"].unique()
    for product in products:
        product_df = df[df["Product"] == product]
        daily_sales = product_df.groupby("Date")["Qty_Sold"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(daily_sales)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast.to_csv(f"{output_dir}/{product}_forecast.csv", index=False)
```

Chatbot classification (illustrative):
- Preprocess text (tokenize, lowercase, remove stopwords).
- Vectorize with CountVectorizer or TF-IDF.
- Train MultinomialNB on labeled intents (e.g., `vegan_request`, `order`, `product_info`).
- Predict intent and use response dictionary or recommendation logic.

---

## Data format & examples

Primary sales dataset (CSV)
- Required columns:
  - Date — YYYY-MM-DD (or parsable by pandas.to_datetime)
  - Product — product name (string)
  - Qty_Sold — integer quantity sold
  - Unit_Price — optional (numeric)
  - Total_Revenue — optional (numeric)

Example:
Date,Product,Qty_Sold,Unit_Price,Total_Revenue
2024-01-01,Croissant,20,2.50,50.00
2024-01-01,Baguette,15,3.00,45.00

Chatbot training dataset (CSV or JSON)
- Example columns:
  - text — user utterance (string)
  - intent — label (string)
- Example rows:
"I want something vegan",vegan_request
"Do you have gluten-free bread?",gluten_free_request

Notes:
- Clean missing and conflicting records before training.
- Add calendar flags (holiday indicator) for better holiday forecasting.
- Balance the training data across intents/products where feasible.

---

## Quickstart (development)

Prerequisites
- Python 3.8+
- pip
- Optional: virtualenv

Install dependencies (example):
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Typical requirements (example)
- pandas
- numpy
- matplotlib
- prophet (or fbprophet)
- scikit-learn
- nltk (or spaCy)
- flask or fastapi (for chatbot API)
- streamlit (or dash) for dashboard

Run forecasting script (example):
```bash
python scripts/forecast.py --input data/sales.csv --output forecast_output --days 14
```

Train chatbot (example):
```bash
python scripts/train_chatbot.py --data data/chatbot_training.csv --model models/chatbot_nb.pkl
```

Start chatbot API (example):
```bash
python app/chatbot_api.py  # launches a local HTTP server
# or
uvicorn app.main:app --reload
```

Start dashboard (example, Streamlit):
```bash
streamlit run app/dashboard.py
```

---

## Typical workflows

Demand forecasting workflow
1. Ingest daily sales CSVs or connect to POS/DB.
2. Clean & aggregate sales per product per date.
3. Train/fit the time-series model for each product (update weekly or daily).
4. Generate forecast files and production recommendations.
5. Publish forecasts to dashboard and notify production staff.

Chatbot workflow
1. Accept user message (web, kiosk, or messaging).
2. Preprocess message and vectorize.
3. Predict intent with the trained classifier.
4. Retrieve recommended products (rules + model + inventory constraints).
5. Present response and optionally create order.

Inventory & manufacturing recommendations
- Translate forecasted demand into ingredient quantities using product-to-ingredient BOM (Bill of Materials).
- Flag items with predicted stockouts or excess inventory for procurement or production adjustments.

---

## Monitoring & evaluation

Key metrics to track:
- Forecast accuracy (MAE, RMSE, MAPE). Target: overall ≥ 85% accuracy on daily/weekly forecasts.
- Waste reduction (%) from production adjustments.
- Stock availability for top N SKUs.
- Chatbot intent accuracy and precision/recall per intent.
- Chatbot response time (target 2–3 seconds).
- Uptime during business hours (target 95%).

Evaluation suggestions:
- Backtest forecasts using rolling-origin evaluation and compare models (Prophet vs. SARIMA vs. ML regressors).
- Use confusion matrix and per-intent F1-scores for the chatbot.

---

## Architecture & components (recommended)

- Data ingestion: ETL scripts or connectors to POS/CSV files.
- Data store: Relational DB (Postgres/SQLite) or cloud storage for CSVs.
- Forecasting engine: Python scripts/services using Prophet and scheduled retraining.
- Chatbot service: Lightweight HTTP service (Flask/FastAPI) exposing endpoints for intent detection and recommendations.
- Dashboard: Streamlit/Dash or a web app showing forecasts, KPI charts, and chatbot logs.
- Orchestration: Cron or GitHub Actions for scheduled training/forecast generation.
- Security: Data encryption in transit, access control to dashboards, compliance with POPIA and local regulations.

---

## Risks & constraints

Risks
- Data quality: Inconsistent or missing historical records can degrade forecast accuracy.
- Adoption: Staff/customers may be hesitant to use the chatbot or trust automated forecasts.
- Integration: Compatibility with existing POS/inventory systems may require custom connectors.
- Security/privacy: Customer data must be handled in compliance with POPIA and other regulations.
- Model performance: External events (holidays, outages, load shedding) can affect accuracy.

Constraints
- Budget limitations: Prefer open-source libraries and free-tier services.
- Time: Academic timelines may limit extensive experimentation or long-term tuning.
- Data availability: Requires sufficiently rich historical data for accurate model training.

Mitigation
- Start with a minimum viable model and incrementally improve.
- Validate models with stakeholders and provide simple override tools for staff.
- Implement logging and manual review steps for early deployments.

---
