import streamlit as st
import pandas as pd
import pickle

# --- LOAD THE SAVED MODEL AND ENCODERS ---
# Use a cache to avoid reloading the model on every interaction, which speeds up the app.
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders from the pickle files."""
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model_data['model'], encoders, model_data['features_names']
    except FileNotFoundError:
        st.error("Model or encoder files not found. Please ensure 'customer_churn_model.pkl' and 'encoders.pkl' are in the same directory.")
        return None, None, None

model, encoders, feature_names = load_model_and_encoders()

# --- APP TITLE AND DESCRIPTION ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Prediction App")
st.write(
    "This app predicts whether a customer is likely to churn based on their account information. "
    "Fill in the details below to get a prediction."
)

# --- CREATE USER INPUT WIDGETS IN THE SIDEBAR ---
st.sidebar.header("Customer Details")

# Create a dictionary to hold user inputs
inputs = {}

# The app creates input fields for each feature the model was trained on.
if feature_names:
    # --- DEMOGRAPHICS ---
    st.sidebar.subheader("Demographics")
    inputs['gender'] = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    inputs['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    inputs['Partner'] = st.sidebar.selectbox("Partner", ['Yes', 'No'])
    inputs['Dependents'] = st.sidebar.selectbox("Dependents", ['Yes', 'No'])

    # --- ACCOUNT INFORMATION ---
    st.sidebar.subheader("Account Information")
    inputs['tenure'] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    inputs['Contract'] = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    inputs['PaperlessBilling'] = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
    inputs['PaymentMethod'] = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    inputs['MonthlyCharges'] = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
    inputs['TotalCharges'] = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1000.0)
    
    # --- SUBSCRIBED SERVICES ---
    st.sidebar.subheader("Subscribed Services")
    inputs['PhoneService'] = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
    inputs['MultipleLines'] = st.sidebar.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    inputs['InternetService'] = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    inputs['OnlineSecurity'] = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    inputs['OnlineBackup'] = st.sidebar.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    inputs['DeviceProtection'] = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    inputs['TechSupport'] = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    inputs['StreamingTV'] = st.sidebar.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    inputs['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])


# --- PREDICTION LOGIC ---
if model and encoders and st.button("Predict Churn"):
    # 1. Convert the user inputs into a pandas DataFrame
    input_df = pd.DataFrame([inputs])

    # 2. Encode the categorical features using the loaded encoders
    # This is a crucial step to match the format of the training data
    for column, encoder in encoders.items():
        try:
            # Use a list for the value to be transformed
            input_df[column] = encoder.transform(input_df[column])
        except Exception as e:
            st.error(f"Error encoding column {column}: {e}")

    # Reorder columns to match the order used during training
    try:
        input_df = input_df[feature_names]
    except KeyError as e:
        st.error(f"A feature is missing from the input: {e}. Please check the feature list.")
    
    # 3. Make a prediction and get probabilities
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # --- DISPLAY THE PREDICTION ---
        st.subheader("Prediction Result")
        if prediction == 1:
            st.warning("Prediction: **Customer will CHURN**")
        else:
            st.success("Prediction: **Customer will NOT CHURN**")

        st.subheader("Prediction Probability")
        # Display probabilities in a more readable format
        prob_df = pd.DataFrame({
            "Outcome": ["Not Churn", "Churn"],
            "Probability": prediction_proba
        })
        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True)

        # Optional: Add a visual gauge/progress bar
        st.progress(prediction_proba[1])
        st.caption(f"Probability of Churn: {prediction_proba[1]:.2%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
