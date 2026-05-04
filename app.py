import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title(" Stock Market Prediction App")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("/workspaces/Stock-Market-Project/Notebook1/stock_data.csv")
    return data

data = load_data()

# Show data
st.subheader("Dataset Preview")
st.write(data.head())

# Use Close column (change if needed)
if 'Stock_1' in data.columns:
    df = data[['Stock_1']]
else:
    df = data.iloc[:, [0]]  # fallback

# Create prediction column
df['Prediction'] = df['Stock_1'].shift(-10)

# Prepare data
X = np.array(df[['Stock_1']][:-10])
y = np.array(df['Prediction'][:-10])

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
st.subheader("Enter Current Price")
input_price = st.number_input("Stock Price", value=100.0)




if st.button("Predict Future Price",key="predict_button_unique"):
    prediction = model.predict([[input_price]])
    st.success(f"Predicted Price: {prediction[0]:.2f}")

    # Add to graph
    fig, ax = plt.subplots()

    ax.plot(df['Stock_1'], label="Actual Price")

    # New input point
    ax.scatter(len(df), input_price, label="Input Price")

    # Predicted point
    ax.scatter(len(df)+1, prediction[0], label="Predicted Price")

    ax.legend()
    st.pyplot(fig)