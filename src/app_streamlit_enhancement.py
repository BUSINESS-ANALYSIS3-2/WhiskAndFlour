import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# Load and preprocess chatbot model
stop_words = set(stopwords.words("english"))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([t for t in tokens if t.isalpha() and t not in stop_words])

training_data = [
    ("I want something vegan", "vegan_request"),
    ("Do you have gluten-free bread?", "gluten_free_request"),
    ("Tell me about croissants", "product_info"),
    ("I’d like to order a birthday cake", "place_order"),
    ("What do you recommend today?", "recommendation"),
]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([preprocess(text) for text, label in training_data])
y_train = [label for text, label in training_data]
model = MultinomialNB()
model.fit(X_train, y_train)

responses = {
    "vegan_request": "We have Vegan Chocolate Tart and Banana Muffins. Would you like to place an order?",
    "gluten_free_request": "Yes! Our gluten-free options include almond flour bread and coconut cookies.",
    "product_info": "Our croissants and baguettes are freshly baked every morning.",
    "place_order": "Great! What would you like to order today?",
    "recommendation": "Our cinnamon rolls and chocolate tarts are trending this week!",
}

# Forecasting function
def forecast_sales(product_name, df, periods=14):
    df = df[df["Product"] == product_name]
    daily = df.groupby("Date")["Qty_Sold"].sum().reset_index()
    daily.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(daily)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# Load sales data
@st.cache_data
def load_data():
    df = pd.read_csv("data/sales_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# Streamlit UI
st.set_page_config(page_title="Whisk & Flour AI", layout="wide")
st.title("🥐 Whisk & Flour Bakery AI Dashboard")

tabs = st.tabs(["📈 Forecasting", "💬 Chatbot"])

# Forecasting Tab
with tabs[0]:
    st.header("Sales Forecasting")
    df = load_data()
    product = st.selectbox("Select a product to forecast", df["Product"].unique())
    days = st.slider("Forecast days", 7, 30, 14)
    if st.button("Generate Forecast"):
        model, forecast = forecast_sales(product, df, days)
        fig = model.plot(forecast)
        st.pyplot(fig)
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days))

# Chatbot Tab
with tabs[1]:
    st.header("Bakery Chatbot Assistant")
    user_input = st.text_input("Ask me anything about our products:")
    if user_input:
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        intent = model.predict(vectorized)[0]
        response = responses.get(intent, "I'm here to help! Could you clarify your request?")
        st.success(response)
