import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load Dataset
# -----------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California House Price Prediction")

st.write(
    """
Predict the **Median House Value** using the California Housing dataset.
Adjust the values below and click **Predict Price**.
"""
)

st.success(f"Model R² Score : {score:.2%}")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
medinc = st.slider(
    "Median Income",
    0.0, 15.0, 4.0
)

houseage = st.slider(
    "House Age",
    1.0, 52.0, 25.0
)

averooms = st.slider(
    "Average Rooms",
    1.0, 15.0, 5.5
)

avebed = st.slider(
    "Average Bedrooms",
    0.5, 5.0, 1.0
)

population = st.slider(
    "Population",
    1, 35000, 1500
)

aveoccup = st.slider(
    "Average Occupancy",
    1.0, 10.0, 3.0
)

latitude = st.slider(
    "Latitude",
    32.0, 42.0, 36.0
)

longitude = st.slider(
    "Longitude",
    -125.0, -114.0, -120.0
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "MedInc":[medinc],
        "HouseAge":[houseage],
        "AveRooms":[averooms],
        "AveBedrms":[avebed],
        "Population":[population],
        "AveOccup":[aveoccup],
        "Latitude":[latitude],
        "Longitude":[longitude]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Predicted House Price")

    st.success(f"${prediction*100000:,.2f}")

    st.progress(min(prediction/5.0,1.0))

st.divider()

st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
}).sort_values("Importance",ascending=False)

st.bar_chart(
    importance.set_index("Feature")
)

st.caption("Developed using Streamlit and Scikit-Learn")