
# 🏡 Housing Price Analysis & Prediction

This project is an **interactive Streamlit web application** for exploring, analyzing, and predicting housing prices. It enables users to upload their datasets, visualize important insights, train machine learning models, and predict housing prices based on selected input features. This tool is designed for **data enthusiasts, students, real estate analysts, and machine learning practitioners**.

---

## 🚀 Features

* 📂 **Upload Custom Dataset** – Accepts CSV files for housing data.
* 🔎 **Exploratory Data Analysis (EDA):**

  * View raw data with filters
  * Summary statistics for numerical features
  * Missing values report
  * Correlation heatmap for feature relationships
  * Distribution plots for numerical features
* 🤖 **Machine Learning Models:**

  * Linear Regression – interpretable baseline model
  * Random Forest Regressor – robust model with feature importance
* 📊 **Model Performance Metrics:**

  * Mean Squared Error (MSE)
  * R² Score
  * Comparison between models
* 🔥 **Feature Importance Visualization** – Identify key drivers of housing price
* 🏠 **Interactive Predictions:**

  * Input custom values for features
  * Get instant housing price predictions

---

## 📊 Example Insights

Using this tool, you can:

* Identify which variables (e.g., number of bedrooms, square footage, location features) most influence housing prices.
* Detect missing values and outliers that may affect predictions.
* Visualize correlations between features (e.g., size vs. price, location vs. price).
* Compare performance of simple vs. complex models.
* Make data-driven predictions for new or hypothetical houses.

---

## 🛠️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/housing-price-analysis.git
   cd housing-price-analysis
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run housing_price_app.py
```

Upload your housing dataset (CSV) and start analyzing!


---

## 📦 Dependencies

* streamlit
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

(Already included in `requirements.txt`)

---

## 📈 Use Cases

* **Real Estate Companies:** Quickly analyze market trends.
* **Students & Researchers:** Learn about regression and model evaluation.
* **Data Analysts:** Generate insights on housing datasets.
* **Buyers & Sellers:** Estimate fair prices based on property attributes.

---

## 💡 Future Improvements

* Add more ML models (XGBoost, LightGBM, Neural Networks)
* Hyperparameter tuning with GridSearchCV
* Advanced feature engineering (categorical encoding, outlier removal)
* Geospatial visualization (maps for location-based prices)
* Export trained model for reuse
* Deploy to Streamlit Cloud / Heroku for public access

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---
