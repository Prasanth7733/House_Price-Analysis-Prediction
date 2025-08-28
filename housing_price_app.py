import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# App Title
# ----------------------------
st.title("🏡 Housing Price Analysis & Prediction")

# ----------------------------
# Upload File
# ----------------------------
uploaded_file = st.file_uploader("Upload your housing dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data successfully loaded!")
    
    # Show data
    if st.checkbox("Show raw data"):
        st.write(df.head())

    # ----------------------------
    # EDA Section
    # ----------------------------
    st.subheader("Exploratory Data Analysis")
    st.write("Shape of dataset:", df.shape)

    if st.checkbox("Show summary statistics"):
        st.write(df.describe())

    if st.checkbox("Show missing values"):
        st.write(df.isnull().sum())

    # Correlation heatmap
    if st.checkbox("Show correlation heatmap"):
        plt.figure(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # ----------------------------
    # Model Training
    # ----------------------------
    st.subheader("Model Training")
    target_col = st.selectbox("Select Target Column (Price)", df.columns)
    feature_cols = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col])

    if st.button("Train Model"):
        X = df[feature_cols]
        y = df[target_col]

        # Handle missing values
        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        lin_reg = LinearRegression()
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

        lin_reg.fit(X_train, y_train)
        rf_reg.fit(X_train, y_train)

        # Predictions
        y_pred_lr = lin_reg.predict(X_test)
        y_pred_rf = rf_reg.predict(X_test)

        # Evaluation
        st.write("### Linear Regression Performance")
        st.write("MSE:", mean_squared_error(y_test, y_pred_lr))
        st.write("R²:", r2_score(y_test, y_pred_lr))

        st.write("### Random Forest Performance")
        st.write("MSE:", mean_squared_error(y_test, y_pred_rf))
        st.write("R²:", r2_score(y_test, y_pred_rf))

        # Feature importance
        st.write("### Feature Importance (Random Forest)")
        feat_imp = pd.Series(rf_reg.feature_importances_, index=feature_cols)
        st.bar_chart(feat_imp)

    # ----------------------------
    # Prediction Section
    # ----------------------------
    st.subheader("Make Prediction")
    if feature_cols:
        input_data = {}
        for col in feature_cols:
            val = st.number_input(f"Enter value for {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = val

        if st.button("Predict Price"):
            input_df = pd.DataFrame([input_data])
            prediction = rf_reg.predict(input_df)[0]
            st.success(f"Predicted House Price: {prediction:.2f}")

else:
    st.info("👆 Upload a dataset to get started!")
