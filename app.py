import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Retail DSS", layout="wide", page_icon="🛍️")


# --- LOAD MODELS FUNCTION ---
@st.cache_resource
def load_models():
    models = {}
    try:
        # Check paths and load models
        # Ensure your models are saved in the 'models' directory
        models['kmeans'] = joblib.load('models/kmeans_model.pkl')
        models['scaler'] = joblib.load('models/scaler_rfm.pkl')
        models['classification'] = joblib.load('models/classification_model.pkl')
        models['sentiment'] = joblib.load('models/sentiment_model.pkl')
        models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
        # models['encoder'] = joblib.load('models/gender_encoder.pkl') # Uncomment if you use a label encoder file
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.warning("Please ensure all .pkl files are located in the 'models' directory.")
    return models


models = load_models()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
st.sidebar.title("Smart Retail DSS")
menu = st.sidebar.radio(
    "Select Module:",
    ["🏠 Home", "🔮 Sales Forecasting", "👥 Customer Segmentation", "🛍️ Product Recommendation", "💬 Review Analysis (NLP)"]
)

st.sidebar.info("Developed by: Berkay, Aras, Güner")

# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if menu == "🏠 Home":
    st.title("🛍️ Intelligent Decision Support System")
    st.markdown("""
    Welcome to the Smart Retail DSS. This system leverages **Artificial Intelligence and Data Analytics** to optimize retail operations.

    ### 🚀 Available Modules:
    - **Sales Forecasting:** Predict future revenue using Time Series Analysis (ARIMA).
    - **Customer Segmentation:** Group customers into VIP, Loyal, or At-Risk categories (RFM & K-Means).
    - **Product Recommendation:** Predict the most suitable product category for a user (Random Forest).
    - **Review Analysis:** Automatically classify customer feedback sentiment (NLP).
    """)

    # Dashboard KPI Cards (Dummy Data for Demo)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", "1,245", "+12%")
    col2.metric("Monthly Revenue", "$45,200", "-5%")
    col3.metric("Customer Satisfaction", "4.8/5.0", "+0.2")

    st.image("https://www.insider.com/public/assets/img/solutions/retail/retail-hero.png", use_column_width=True)

# ==============================================================================
# PAGE 2: SALES FORECASTING (ARIMA)
# ==============================================================================
elif menu == "🔮 Sales Forecasting":
    st.title("📈 Sales Forecasting Module")
    st.subheader("Time Series Analysis with ARIMA")

    st.info("This module analyzes historical sales data to forecast demand for the upcoming weeks.")

    # Generate dummy forecast data for visualization purposes
    days = pd.date_range(start='2024-01-01', periods=30)
    sales = np.random.randint(1000, 5000, size=30)
    forecast = np.random.randint(4000, 6000, size=5)  # 5 days forecast

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, sales, label='Historical Sales', color='blue')
    ax.plot(pd.date_range(start=days[-1], periods=5), forecast, label='AI Forecast', color='red', linestyle='--')
    ax.set_title("Sales Prediction (Next 5 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.success("💡 Recommendation: High demand expected next week. Increase stock for 'Electronics' category by 10%.")

# ==============================================================================
# PAGE 3: CUSTOMER SEGMENTATION (CLUSTERING)
# ==============================================================================
elif menu == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation (RFM Analysis)")

    st.write("Enter customer transaction details to identify their behavioral segment.")

    col1, col2, col3 = st.columns(3)
    recency = col1.number_input("Recency (Days since last purchase)", min_value=0, value=10)
    frequency = col2.number_input("Frequency (Total number of purchases)", min_value=1, value=5)
    monetary = col3.number_input("Monetary (Total amount spent $)", min_value=0.0, value=500.0)

    if st.button("Identify Segment"):
        if 'kmeans' in models and 'scaler' in models:
            # Prepare and scale data
            input_data = np.array([[recency, frequency, monetary]])
            scaled_data = models['scaler'].transform(input_data)

            # Predict cluster
            cluster = models['kmeans'].predict(scaled_data)[0]

            # Map cluster ID to Segment Name
            segment_names = {
                0: "🏆 VIP Customer (High Value)",
                1: "💎 Loyal Customer (Frequent)",
                2: "⚠️ At-Risk Customer (High Past Spending)",
                3: "💤 Lost / Low Value"
            }
            result = segment_names.get(cluster, "Unknown Segment")

            st.success(f"Customer Segment: **{result}**")

            if cluster == 0:
                st.balloons()
        else:
            st.error("Model files not found! Please check the 'models' folder.")

# ==============================================================================
# PAGE 4: PRODUCT RECOMMENDATION (CLASSIFICATION)
# ==============================================================================
elif menu == "🛍️ Product Recommendation":
    st.title("🛍️ Personalized Product Recommendation")

    st.write("Predict the most suitable product category based on customer demographics.")

    col1, col2 = st.columns(2)
    gender = col1.selectbox("Gender", ["Female", "Male"])
    age = col2.slider("Age", 18, 80, 25)

    col3, col4 = st.columns(2)
    income = col3.slider("Annual Income ($)", 10000, 150000, 50000)
    spending_score = col4.slider("Spending Score (1-100)", 1, 100, 50)

    # Encoding Gender: Assuming model was trained with Female=0, Male=1
    gender_code = 1 if gender == "Male" else 0

    if st.button("Get Recommendation"):
        if 'classification' in models:
            # Prepare input vector: [Age, Gender_Code, Annual_Income, Spending_Score]
            input_vector = np.array([[age, gender_code, income, spending_score]])

            prediction = models['classification'].predict(input_vector)[0]

            st.info(f"Recommended Category: **{prediction}**")
        else:
            st.error("Classification model not found.")

# ==============================================================================
# PAGE 5: REVIEW ANALYSIS (NLP)
# ==============================================================================
elif menu == "💬 Review Analysis (NLP)":
    st.title("💬 Voice of Customer (Sentiment Analysis)")

    st.write("Enter a customer review to analyze the sentiment automatically.")

    user_review = st.text_area("Customer Review:", "The product quality is amazing and delivery was super fast!")

    if st.button("Analyze Sentiment"):
        if 'sentiment' in models and 'vectorizer' in models:
            # 1. Vectorize text
            text_vector = models['vectorizer'].transform([user_review]).toarray()

            # 2. Predict sentiment
            sentiment = models['sentiment'].predict(text_vector)[0]

            # 3. Display Result
            if sentiment == "Positive":
                st.success("Sentiment: 😊 POSITIVE")
            elif sentiment == "Negative":
                st.error("Sentiment: 😡 NEGATIVE")
            else:
                st.warning("Sentiment: 😐 NEUTRAL")
        else:
            st.error("NLP Model or Vectorizer not found.")