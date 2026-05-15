import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import numpy as np
# ======================================
# Page Configuration
# ======================================

st.set_page_config(
    page_title="Industrial Failure Prediction System",
    page_icon="⚙️",
    layout="wide"
)
# ======================================
# Load Saved Model
# ======================================

with open('../models/xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('../models/feature_columns.pkl', 'rb') as file:
    feature_columns = pickle.load(file)


# ======================================
# SHAP Explainer
# ======================================

explainer = shap.Explainer(model)

# ======================================
# Page Title
# ======================================

st.title("⚙️ Industrial Equipment Monitoring System")

st.markdown("""
### AI-Powered Predictive Maintenance Dashboard

Real-time industrial equipment monitoring using:
- XGBoost Machine Learning
- Operational Sensor Analytics
- Explainable AI (SHAP)
- Predictive Maintenance Intelligence
""")

# ======================================
# Sidebar Inputs
# ======================================

st.sidebar.title("Machine Configuration")

st.sidebar.markdown("""
Adjust operational sensor values to evaluate
equipment failure risk.
""")

st.sidebar.divider()

st.sidebar.header("Sensor Inputs")

# Air Temperature
air_temp = st.sidebar.slider(
    "Air Temperature (K)",
    min_value=295.0,
    max_value=305.0,
    value=300.0
)

# Process Temperature
process_temp = st.sidebar.slider(
    "Process Temperature (K)",
    min_value=305.0,
    max_value=315.0,
    value=310.0
)

# Rotational Speed
rpm = st.sidebar.slider(
    "Rotational Speed (RPM)",
    min_value=1100,
    max_value=3000,
    value=1500
)

# Torque
torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

# Tool Wear
tool_wear = st.sidebar.slider(
    "Tool Wear (min)",
    min_value=0,
    max_value=250,
    value=100
)

# Machine Type
machine_type = st.sidebar.selectbox(
    "Machine Type",
    ['L', 'M', 'H']
)


# ======================================
# Display Current Inputs
# ======================================

st.subheader("Current Machine Inputs")

st.write(f"Air Temperature: {air_temp} K")
st.write(f"Process Temperature: {process_temp} K")
st.write(f"Rotational Speed: {rpm} RPM")
st.write(f"Torque: {torque} Nm")
st.write(f"Tool Wear: {tool_wear} min")
st.write(f"Machine Type: {machine_type}")



# ======================================
# Encode Machine Type
# ======================================

type_L = 1 if machine_type == 'L' else 0
type_M = 1 if machine_type == 'M' else 0

# ======================================
# Feature Engineering
# ======================================

temp_diff = process_temp - air_temp

power_proxy = rpm * torque

wear_torque_interaction = tool_wear * torque

speed_temp_interaction = rpm * process_temp

# ======================================
# Create Input Data
# ======================================

input_data = pd.DataFrame({
    'Air temperature _K_': [air_temp],
    'Process temperature _K_': [process_temp],
    'Rotational speed _rpm_': [rpm],
    'Torque _Nm_': [torque],
    'Tool wear _min_': [tool_wear],
    'Type_L': [type_L],
    'Type_M': [type_M],
    'Temp_diff': [temp_diff],
    'Power_proxy': [power_proxy],
    'Wear_Torque_Interaction': [wear_torque_interaction],
    'Speed_Temp_Interaction': [speed_temp_interaction]

})

input_data = input_data[feature_columns]

# ======================================
# Model Prediction
# ======================================

prediction = model.predict(input_data)[0]

prediction_probability = model.predict_proba(input_data)[0][1]

st.sidebar.divider()

st.sidebar.subheader("Current Risk Status")

st.sidebar.write(
    f"Failure Probability: {prediction_probability:.2%}"
)
# ======================================
# SHAP Values
# ======================================

shap_values = explainer(input_data)


# ======================================
# Prediction Results
# ======================================

st.header("Operational Risk Monitoring")

col1, col2 = st.columns(2)

# Failure Probability Card
with col1:

    st.metric(
        label="Failure Probability",
        value=f"{prediction_probability:.2%}"
    )

# Risk Gauge Color Logic

if prediction_probability < 0.30:
    st.success("System Stability: GOOD")

elif prediction_probability < 0.70:
    st.warning("System Stability: MODERATE RISK")

else:
    st.error("System Stability: CRITICAL")

# Risk Progress Bar

st.progress(float(prediction_probability))


# ======================================
# Risk Status
# ======================================

if prediction_probability < 0.30:

    st.success("Low Failure Risk")

elif prediction_probability < 0.70:

    st.warning("Moderate Failure Risk")

else:

    st.error("High Failure Risk Detected")

# ======================================
# Engineered Features
# ======================================

st.subheader("Engineered Operational Features")

engineered_features = pd.DataFrame({
    'Feature': [
        'Temperature Difference',
        'Power Proxy',
        'Wear-Torque Interaction',
        'Speed-Temperature Interaction'
    ],
    'Value': [
        temp_diff,
        power_proxy,
        wear_torque_interaction,
        speed_temp_interaction
    ]
})

st.dataframe(engineered_features)

# ======================================
# Raw Input Data
# ======================================

st.subheader("Current Sensor Readings")

st.dataframe(input_data)

# ======================================
# Sensor Visualization
# ======================================

st.subheader("Sensor Monitoring Overview")

sensor_df = pd.DataFrame({
    'Sensor': [
        'Air Temp',
        'Process Temp',
        'RPM',
        'Torque',
        'Tool Wear'
    ],
    'Value': [
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear
    ]
})

fig = px.bar(
    sensor_df,
    x='Sensor',
    y='Value',
    title='Live Sensor Monitoring',
    text='Value'
)

fig.update_layout(
    template='plotly_dark',
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================
# Simulated Historical Monitoring
# ======================================

st.header("Historical Sensor Trend Monitoring")

# Simulated timeline
time_steps = np.arange(1, 51)

# Simulated sensor history
rpm_history = rpm + np.random.normal(0, 50, 50)

torque_history = torque + np.random.normal(0, 3, 50)

temp_history = process_temp + np.random.normal(0, 0.5, 50)

# Create dataframe
history_df = pd.DataFrame({
    'Time': time_steps,
    'RPM': rpm_history,
    'Torque': torque_history,
    'Process Temperature': temp_history
})

st.line_chart(
    history_df.set_index('Time')
)
# ======================================
# SHAP Explainability
# ======================================

st.subheader("AI Explainability (SHAP)")

st.write(
    "Feature contributions influencing current prediction."
)
st.divider()

fig, ax = plt.subplots(figsize=(10, 5))

shap.plots.waterfall(
    shap_values[0],
    show=False
)

# ======================================
# Maintenance Report Generation
# ======================================

st.header("Maintenance Report")

# Risk level logic
if prediction_probability < 0.30:
    risk_level = "LOW"

elif prediction_probability < 0.70:
    risk_level = "MODERATE"

else:
    risk_level = "HIGH"

# Create report dataframe
report_df = pd.DataFrame({

    'Metric': [
        'Air Temperature',
        'Process Temperature',
        'Rotational Speed',
        'Torque',
        'Tool Wear',
        'Machine Type',
        'Failure Probability',
        'Risk Level'
    ],

    'Value': [
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        machine_type,
        f"{prediction_probability:.2%}",
        risk_level
    ]
})

# Display report table
st.dataframe(report_df)


# Convert report to CSV
csv = report_df.to_csv(index=False)

# Download button
st.download_button(
    label="Download Maintenance Report",
    data=csv,
    file_name='maintenance_report.csv',
    mime='text/csv'
)

st.pyplot(fig)

st.divider()

st.caption(
    "Industrial Equipment Failure Prediction System | Built using XGBoost, SHAP, Streamlit"
)

# ======================================
# Technical Information
# ======================================

with st.expander("View Technical Model Details"):

    st.write("""
    ### Model Information

    - Algorithm: XGBoost Classifier
    - Problem Type: Predictive Maintenance Classification
    - Target: Machine Failure Prediction
    - Explainability: SHAP
    - Feature Engineering:
        - Power Proxy
        - Temperature Difference
        - Torque-Wear Interaction
        - Speed-Temperature Interaction

    ### Operational Goal

    Detect industrial equipment failure risk
    before breakdown occurs.
    """)