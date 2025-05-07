# streamlit_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("wine_quality_classification.csv")

df = load_data()

# Title
st.title("🍷 Wine Quality Analysis")

# Dataset Overview
st.subheader("Dataset Preview")
st.write(df.head())

# Sidebar for feature selection
st.sidebar.header("Customize Plot")
feature = st.sidebar.selectbox("Choose a numeric feature", ['alcohol', 'residual_sugar', 'fixed_acidity', 'density'])

# Distribution Plot
st.subheader(f"Distribution of {feature}")
fig1, ax1 = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax1)
st.pyplot(fig1)

# Boxplot by Quality
st.subheader(f"{feature} vs Wine Quality (Boxplot)")
fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x='quality_label', y=feature, order=['low', 'medium', 'high'], ax=ax2)
st.pyplot(fig2)

# Regression plot: Alcohol vs Residual Sugar by Wine Quality
st.subheader("Alcohol vs Residual Sugar with Regression by Wine Quality")
lm_fig = sns.lmplot(data=df, x='residual_sugar', y='alcohol', hue='quality_label', height=5, aspect=1.5)
st.pyplot(lm_fig.figure)

# Conclusion Text
st.markdown("""
### 📌 Insights Summary
- **Alcohol** level is the best predictor of wine quality.
- **Residual sugar**, **density**, and **fixed acidity** show no clear relationship with quality.
- Wines with **higher alcohol** content are more likely to be labeled as high-quality.

---
""")
