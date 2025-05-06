import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Dashboard", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("wine_quality_classification.csv")

w_df = load_data()

st.title("🍷 Wine Quality Classification Dashboard")
st.markdown("""
Explore the relationship between chemical properties and wine quality.
Dataset: 1,000 wines labeled as **low**, **medium**, or **high** quality.
""")

# Show data toggle
if st.checkbox("Show Raw Data"):
    st.dataframe(w_df)

# Boxplot: Alcohol by quality
st.subheader("Alcohol Content by Wine Quality")
fig1, ax1 = plt.subplots()
sns.boxplot(data=w_df, x='quality_label', y='alcohol', order=['low', 'medium', 'high'], ax=ax1)
st.pyplot(fig1)

# KDE plot: Residual Sugar by quality
st.subheader("Residual Sugar Distribution by Quality")
fig2, ax2 = plt.subplots()
sns.kdeplot(data=w_df, x='residual_sugar', hue='quality_label', fill=True, ax=ax2)
st.pyplot(fig2)

# Scatterplot: Alcohol vs Residual Sugar
st.subheader("Alcohol vs Residual Sugar")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=w_df, x='residual_sugar', y='alcohol', hue='quality_label', ax=ax3)
st.pyplot(fig3)

# Key takeaways
st.markdown("### 🧠 Key Insights")
st.markdown("""
- High quality wines have higher alcohol content.
- Residual sugar varies and is less predictive.
- Wines with high alcohol and low sugar are often higher quality.
""")
