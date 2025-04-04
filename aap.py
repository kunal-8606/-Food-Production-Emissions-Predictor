import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Food Emissions Predictor", layout="centered")
st.title("🌾 Food Production Emissions Predictor")

# File upload
uploaded_file = st.file_uploader("📂 Upload your 'Food_Production.csv' file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("🔍 Raw Data Preview")
    st.write(df.head())

    st.subheader("🧹 Cleaning Data")
    df.dropna(inplace=True)
    if "Eutrophying emissions per 100g protein (gPO₄eq per 100 grams protein)" in df.columns:
        df.drop(columns=["Eutrophying emissions per 100g protein (gPO₄eq per 100 grams protein)"], inplace=True)

    st.success("✅ Data cleaned successfully.")
    st.write(df.head())

    # Label encoding
    st.subheader("🔤 Encoding 'Food product'")
    label_encoder = LabelEncoder()
    df['Food product'] = label_encoder.fit_transform(df['Food product'])

    st.write(df[['Food product']].head())

    # Check if target column exists
    if 'Total_emissions' in df.columns:
        st.subheader("🧠 Training the Model")

        # Features and target
        X = df.drop(columns=["Total_emissions"])
        y = df["Total_emissions"]

        # Split
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

        # Train
        model = LinearRegression()
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)

        # Evaluation
        mae = mean_absolute_error(ytest, ypred)
        mse = mean_squared_error(ytest, ypred)
        r2 = r2_score(ytest, ypred)

        # Convert to percentages
        mean_actual = ytest.mean()
        mae_percent = (mae / mean_actual) * 100
        mse_percent = (mse / (ytest ** 2).mean()) * 100
        r2_percent = r2 * 100

        st.success("✅ Model trained successfully!")

        # Metrics
        st.subheader("📊 Model Evaluation Metrics (in %)")
        st.markdown(f"""
        - **Mean Absolute Error (MAE):** `{mae_percent:.2f}%`
        - **Mean Squared Error (MSE):** `{mse_percent:.2f}%`
        - **R² Score:** `{r2_percent:.2f}%`
        """)

        # Show predictions
        st.subheader("🔍 First 10 Predictions")
        results = pd.DataFrame({
            'Actual': ytest[:10].values,
            'Predicted': ypred[:10]
        })
        st.write(results)

        # User input
        st.subheader("🔮 Try Your Own Input")
        input_data = {}
        for col in X.columns:
            if col == "Food product":
                input_data[col] = st.number_input(f"Enter encoded 'Food product' (0 - {df['Food product'].max()}):", 
                                                  min_value=0, max_value=int(df['Food product'].max()), value=0)
            else:
                input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

        if st.button("Predict Emissions"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            st.success(f"🌱 Estimated Total Emissions: **{prediction:.2f} kg CO₂eq**")
    else:
        st.error("❌ Column 'Total_emissions' not found in dataset.")
else:
    st.info("⬆️ Upload a CSV file to get started.")
