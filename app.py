# app.py
# Streamlit Car Resale Price Predictor using your specified CSV columns
# Syllabus module integration comments: [M1], [M2], [M3], [M4], [M5]

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# [M1] Module 1: Data Understanding & Descriptive Statistics
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("car_data.csv")
        df.columns = df.columns.str.strip()  # Remove accidental spaces
        # Check for required columns
        required = {'car_name','brand','model','vehicle_age','km_driven','seller_type',
                    'fuel_type','transmission_type','mileage','engine','max_power','seats','selling_price'}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing columns in dataset: {', '.join(missing)}")
            st.stop()
        return df
    except FileNotFoundError:
        st.error("Dataset file 'car_data.csv' not found!")
        st.stop()

df = load_data()

# [M2] Module 2: Feature Engineering & Data Preprocessing
def preprocess_data(df):
    df = df.copy()
    # Handle missing values
    df = df.dropna()
    # Convert mileage, engine, max_power to numeric (remove non-numeric chars)
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    return df

df = preprocess_data(df)

# [M5] Module 5: Clustering
def add_cluster_features(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['usage_cluster'] = kmeans.fit_predict(df[['km_driven', 'vehicle_age']])
    return df, kmeans

df, kmeans = add_cluster_features(df)

# [M2] Feature selection for modeling
numerical_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'usage_cluster']
categorical_features = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)
    ]
)

# [M3] & [M4] Models
models = {
    "Linear Regression": Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    "Decision Tree": Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))
    ]),
    "Bayesian Regression": Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', BayesianRidge())
    ]),
    "Neural Network": Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42, early_stopping=True))
    ])
}

X = df[numerical_features + categorical_features]
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for model in models.values():
    model.fit(X_train, y_train)

# [M5] RL-inspired adjustment
class PriceAdjuster:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
    def adjust_price(self, predicted_price, cluster):
        state = str(cluster)
        if state not in self.q_table:
            self.q_table[state] = {'up': 0, 'down': 0}
        if np.random.random() < self.epsilon:
            adjustment = np.random.choice(['up', 'down'])
        else:
            adjustment = max(self.q_table[state], key=self.q_table[state].get)
        if adjustment == 'up':
            final_price = predicted_price * 1.05
            reward = 1 if final_price < df['selling_price'].quantile(0.75) else -1
        else:
            final_price = predicted_price * 0.95
            reward = 1 if final_price > df['selling_price'].quantile(0.25) else -1
        self.q_table[state][adjustment] += self.learning_rate * (
            reward + self.discount_factor * max(self.q_table[state].values()) - self.q_table[state][adjustment]
        )
        return final_price

price_adjuster = PriceAdjuster()

# Streamlit UI
st.title("🚗 Car Resale Value Predictor")
st.markdown("Predict your car's resale value using advanced ML techniques!")

# Sidebar: User input (use your columns)
st.sidebar.header("🚘 Enter Car Specifications")
brand = st.sidebar.selectbox("Brand", sorted(df['brand'].unique()))
model = st.sidebar.selectbox("Model", sorted(df[df['brand']==brand]['model'].unique()))
vehicle_age = st.sidebar.slider("Vehicle Age (years)", int(df['vehicle_age'].min()), int(df['vehicle_age'].max()), 5)
km_driven = st.sidebar.slider("Kilometers Driven", int(df['km_driven'].min()), int(df['km_driven'].max()), 50000, step=1000)
seller_type = st.sidebar.selectbox("Seller Type", df['seller_type'].unique())
fuel_type = st.sidebar.selectbox("Fuel Type", df['fuel_type'].unique())
transmission_type = st.sidebar.selectbox("Transmission", df['transmission_type'].unique())
mileage = st.sidebar.number_input("Mileage (kmpl)", float(df['mileage'].min()), float(df['mileage'].max()), float(df['mileage'].mean()))
engine = st.sidebar.number_input("Engine (CC)", float(df['engine'].min()), float(df['engine'].max()), float(df['engine'].mean()))
max_power = st.sidebar.number_input("Max Power (bhp)", float(df['max_power'].min()), float(df['max_power'].max()), float(df['max_power'].mean()))
seats = st.sidebar.selectbox("Seats", sorted(df['seats'].dropna().unique()))

# Get cluster for user input
user_cluster = kmeans.predict([[km_driven, vehicle_age]])[0]

model_choice = st.selectbox("Select Prediction Model", list(models.keys()))

if st.button("Predict Resale Price"):
    input_data = pd.DataFrame([[
        vehicle_age, km_driven, mileage, engine, max_power, seats, user_cluster,
        brand, model, seller_type, fuel_type, transmission_type
    ]], columns=numerical_features + categorical_features)
    base_price = models[model_choice].predict(input_data)[0]
    final_price = price_adjuster.adjust_price(base_price, user_cluster)
    st.success(f"Base Prediction: ₹{base_price:,.2f}")
    st.success(f"RL-Adjusted Price: ₹{final_price:,.2f}")
    st.write(f"Cluster: {user_cluster}")
    st.write(f"Q-values: {price_adjuster.q_table[str(user_cluster)]}")

# [M1] Data Visualizations
st.subheader("📈 Data Insights (Module 1)")
tab1, tab2, tab3 = st.tabs(["Price Distribution", "Feature Relationships", "Cluster Analysis"])

with tab1:
    fig = plt.figure(figsize=(10,6))
    sns.histplot(df['selling_price'], kde=True)
    plt.title("Price Distribution (Log Scale)")
    plt.xscale('log')
    st.pyplot(fig)

with tab2:
    fig = plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='km_driven', y='selling_price', hue='fuel_type')
    plt.title("Price vs Kilometers Driven")
    st.pyplot(fig)

with tab3:
    fig = plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='usage_cluster', y='selling_price')
    plt.title("Price Distribution Across Clusters")
    st.pyplot(fig)

# [M3/M4] Model Comparison
st.subheader("🤖 Model Performance (Module 3 & 4)")
test_scores = {name: model.score(X_test, y_test) for name, model in models.items()}
st.bar_chart(pd.DataFrame(test_scores, index=["R² Score"]))

# [M4] Neural Network Details
if model_choice == "Neural Network":
    st.subheader("🧠 Neural Network Details (Module 4)")
    st.write("**Network Architecture:**")
    st.write(models["Neural Network"].named_steps['regressor'].hidden_layer_sizes)
    st.write("**Training Loss Curve:**")
    fig = plt.figure(figsize=(10,4))
    plt.plot(models["Neural Network"].named_steps['regressor'].loss_curve_)
    plt.title("Training Loss Progress")
    st.pyplot(fig)
