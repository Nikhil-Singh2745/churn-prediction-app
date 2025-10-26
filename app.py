import streamlit as st
import pandas as pd
import pickle
import sqlite3
import datetime
import json
import numpy as np
import yaml  # For login
from yaml.loader import SafeLoader  # For login
import streamlit_authenticator as stauth  # For login
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import base64  # For CSV download

# --- DATABASE CONFIGURATION ---
DB_NAME = "predictions.db"
DATA_URL = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# --- DATABASE HELPER FUNCTIONS (No Changes) ---
@st.cache_resource
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                inputs_json TEXT,
                prediction TEXT,
                churn_probability REAL
            )
        """)
        conn.commit()
    except Exception as e:
        st.error(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def log_prediction(inputs_dict, prediction_label, prediction_proba):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inputs_str = json.dumps(inputs_dict)
        c.execute("""
            INSERT INTO predictions (timestamp, inputs_json, prediction, churn_probability)
            VALUES (?, ?, ?, ?)
        """, (timestamp, inputs_str, prediction_label, prediction_proba))
        conn.commit()
    except Exception as e:
        st.warning(f"Error logging prediction to database: {e}")
    finally:
        if conn:
            conn.close()

def view_predictions():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT timestamp, prediction, churn_probability, inputs_json FROM predictions ORDER BY timestamp DESC", conn)
        return df
    except Exception as e:
        st.error(f"Error reading from database: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# --- DATA LOADING & MODEL LOADING (No Changes) ---
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
        df = df.dropna(subset=["TotalCharges"])
        return df
    except FileNotFoundError:
        st.error(f"Error: {DATA_URL} file not found. Please make sure it's in the same directory.")
        return None

@st.cache_resource
def load_model_and_encoders():
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        if 'model' not in model_data or 'features_names' not in model_data:
            st.error("Model file 'customer_churn_model.pkl' is not in the expected format.")
            return None, None, None
        return model_data['model'], encoders, model_data['features_names']
    except FileNotFoundError:
        st.error("Model or encoder files not found. Please ensure 'customer_churn_model.pkl' and 'encoders.pkl' are in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

# --- NEW HELPER FUNCTIONS ---

def preprocess_features(df_input, encoders, feature_names):
    """
    Preprocesses a DataFrame (either single input or batch)
    using the loaded encoders and feature list.
    """
    df = df_input.copy()
    
    # Handle TotalCharges - convert to numeric, fill NaNs (e.g., with 0)
    # Filling with 0 is safer for prediction than dropping rows in a batch
    if 'TotalCharges' in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
    
    # Apply encoders
    for column, encoder in encoders.items():
        if column in df.columns:
            try:
                df[column] = encoder.transform(df[column])
            except Exception as e:
                st.warning(f"Could not encode column {column}: {e}. Skipping.")
    
    # Reorder columns to match model's expected input
    try:
        df = df[feature_names]
    except KeyError as e:
        st.error(f"Feature mismatch: {e}. Ensure uploaded CSV has all required columns.")
        return None
        
    return df

def get_table_download_link(df, filename, link_text):
    """Generates a link to download a pandas DataFrame as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encode
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


# --- MAIN APPLICATION ---

def main():
    st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
    
    # --- 1. LOGIN (Request 3) ---
    try:
        config = {
            'credentials': {
                'usernames': {
                    st.secrets["credentials"]["usernames"]["jsmith"]["username"]: {
                        'email': st.secrets["credentials"]["usernames"]["jsmith"]["email"],
                        'name': st.secrets["credentials"]["usernames"]["jsmith"]["name"],
                        'password': st.secrets["credentials"]["usernames"]["jsmith"]["password"]
                    },
                    st.secrets["credentials"]["usernames"]["rdoe"]["username"]: {
                        'email': st.secrets["credentials"]["usernames"]["rdoe"]["email"],
                        'name': st.secrets["credentials"]["usernames"]["rdoe"]["name"],
                        'password': st.secrets["credentials"]["usernames"]["rdoe"]["password"]
                    }
                }
            },
            'cookie': {
                'expiry_days': st.secrets["cookie"]["expiry_days"],
                'key': st.secrets["cookie"]["key"],
                'name': st.secrets["cookie"]["name"]
            }
        }
    except KeyError as e:
        st.error(f"A key is missing from your Streamlit secrets: {e}. Please check your secrets.toml.")
        return
    except Exception as e:
        st.error(f"Error loading secrets: {e}")
        return

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    
    # Render the login form
    authenticator.login(location='main')

    # Get the authentication status, name, and username from st.session_state
    # The authenticator object writes its state to st.session_state
    authentication_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    # Show the demo credentials IF the user is not yet logged in
    if not st.session_state.get("authentication_status"):
        st.info(
            """
            **Demo Credentials for Recruiters** - **Username:** `test1`
            - **Password:** `pass1` 
            """
        )

    # --- AUTHENTICATION CHECK ---
    if st.session_state.get("authentication_status") is False:
        st.error('Username/password is incorrect')
    elif st.session_state.get("authentication_status") is None:
        st.warning('Please enter your username and password')
    
    # --- APP RUNS ONLY IF AUTHENTICATION IS SUCCESSFUL ---
    elif st.session_state.get("authentication_status"):
        # --- Load Model, Encoders, and Data ---
        model, encoders, feature_names = load_model_and_encoders()
        raw_df = load_data(DATA_URL)
        
        # --- Initialize Database Table ---
        if feature_names:
            init_db()

        st.sidebar.write(f'Welcome *{name}*')
        st.title("Customer Churn Prediction App 🚀")
        
        # This logout call should be correct
        authenticator.logout(location='sidebar')

        # --- 1. DASHBOARD VIEW (Request 1 from last time) ---
        if raw_df is not None:
            st.header("📊 Dataset Dashboard")
            if st.checkbox("Show Summary Analytics Dashboard"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Customers", f"{len(raw_df):,}")
                churn_rate = (raw_df['Churn'].value_counts(normalize=True)['Yes'] * 100)
                col2.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
                avg_charge = raw_df['MonthlyCharges'].mean()
                col3.metric("Avg. Monthly Charge", f"${avg_charge:.2f}")

                st.subheader("Churn Count by Contract Type")
                churn_by_contract = raw_df[raw_df['Churn'] == 'Yes']['Contract'].value_counts()
                st.bar_chart(churn_by_contract, color="#ff4b4b")

        # --- 2. NEW: MODEL PERFORMANCE ON FULL DATASET (Request 1) ---
        st.header("📈 Model Performance")
        with st.expander("Show Confusion Matrix (on Full Dataset)"):
            if model and raw_df is not None and encoders and feature_names:
                try:
                    # Prepare full dataset for prediction
                    X_full = raw_df.drop(['customerID', 'Churn'], axis=1)
                    y_true_labels = raw_df['Churn']
                    y_true = y_true_labels.map({'Yes': 1, 'No': 0})
                    
                    # Preprocess features
                    X_full_processed = preprocess_features(X_full, encoders, feature_names)
                    
                    if X_full_processed is not None:
                        # Make predictions
                        y_pred = model.predict(X_full_processed)
                        
                        # Calculate confusion matrix
                        cm = confusion_matrix(y_true, y_pred)
                        
                        # Plot
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=['Not Churn', 'Churn'], 
                                    yticklabels=['Not Churn', 'Churn'])
                        ax.set_xlabel('Predicted Label')
                        ax.set_ylabel('True Label')
                        ax.set_title('Confusion Matrix on Full Dataset')
                        st.pyplot(fig)
                        
                        st.write("This matrix shows how the model performs on the *entire* dataset it was (likely) trained on. It helps identify if the model is biased towards one class (e.g., predicting 'Not Churn' too often).")
                except Exception as e:
                    st.error(f"Could not generate confusion matrix: {e}")
            else:
                st.info("Model, encoders, or raw data not loaded correctly.")


        st.divider()

        # --- 3. PREDICTION UI ---
        
        # --- SIDEBAR FOR USER INPUT ---
        st.sidebar.header("Customer Details (Single Prediction)")
        inputs = {}
        if feature_names:
            # (Input fields remain unchanged from the previous version)
            # ... [Demographics, Account Info, Services sections] ...
            st.sidebar.subheader("Demographics")
            inputs['gender'] = st.sidebar.selectbox("Gender", ['Male', 'Female'])
            inputs['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            inputs['Partner'] = st.sidebar.selectbox("Partner", ['Yes', 'No'])
            inputs['Dependents'] = st.sidebar.selectbox("Dependents", ['Yes', 'No'])

            st.sidebar.subheader("Account Information")
            inputs['tenure'] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
            inputs['Contract'] = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            inputs['PaperlessBilling'] = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
            inputs['PaymentMethod'] = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            inputs['MonthlyCharges'] = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0, step=0.05)
            inputs['TotalCharges'] = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0, step=0.1)

            st.sidebar.subheader("Services Subscribed")
            inputs['PhoneService'] = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
            inputs['MultipleLines'] = st.sidebar.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
            inputs['InternetService'] = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            inputs['OnlineSecurity'] = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
            inputs['OnlineBackup'] = st.sidebar.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
            inputs['DeviceProtection'] = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
            inputs['TechSupport'] = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
            inputs['StreamingTV'] = st.sidebar.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
            inputs['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
        else:
            st.sidebar.error("Model not loaded. Cannot create input fields.")


        st.header("🔮 New Single Customer Prediction")
        if model and encoders and feature_names:
            if st.button("Predict Churn", type="primary"):
                input_df = pd.DataFrame([inputs])
                input_processed = preprocess_features(input_df, encoders, feature_names)
                
                if input_processed is not None:
                    prediction = model.predict(input_processed)[0]
                    prediction_proba = model.predict_proba(input_processed)[0]
                    churn_probability = prediction_proba[1]

                    st.subheader("Prediction Result")
                    if prediction == 1:
                        prediction_label = "Churn"
                        st.warning(f"Prediction: **Customer will CHURN** (Probability: {churn_probability:.2%})")
                    else:
                        prediction_label = "Not Churn"
                        st.success(f"Prediction: **Customer will NOT CHURN** (Probability: {1-churn_probability:.2%})")
                    
                    st.progress(churn_probability)
                    st.caption(f"Probability of Churn: {churn_probability:.2%}")
                    
                    log_prediction(inputs, prediction_label, churn_probability)
                    st.toast("Prediction logged to database!")

                    # --- Retention Suggestion (Unchanged) ---
                    st.subheader("💡 Customer Retention Suggestion")
                    if prediction == 1:
                        if inputs['tenure'] <= 12 and inputs['MonthlyCharges'] >= 70:
                            st.info("High-risk customer (new, high-spend). **Suggestion:** Offer a 1-year contract discount or a free service upgrade.")
                        elif inputs['Contract'] == 'Month-to-month':
                            st.info("This customer is on a flexible Month-to-Month contract. **Suggestion:** Propose a 1-year contract at a reduced rate.")
                        else:
                            st.info("Review support history for unresolved issues or offer a small loyalty bonus.")
                    else:
                        st.success("✅ Customer likely satisfied. No immediate retention action needed.")

        st.divider()

        # --- 4. NEW: BATCH PREDICTION (Request 2) ---
        st.header("📤 Batch Prediction (Upload CSV)")
        uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])
        
        if uploaded_file is not None and model and encoders and feature_names:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Snippet:")
                st.dataframe(batch_df.head())
                
                # Keep a copy of original columns for output
                original_cols = batch_df.columns.tolist()
                
                # Preprocess the batch data
                batch_processed = preprocess_features(batch_df, encoders, feature_names)
                
                if batch_processed is not None:
                    # Make predictions
                    batch_df['Churn_Prediction_Label'] = model.predict(batch_processed)
                    batch_df['Churn_Prediction_Label'] = batch_df['Churn_Prediction_Label'].map({1: 'Churn', 0: 'Not Churn'})
                    batch_df['Churn_Probability'] = model.predict_proba(batch_processed)[:, 1]
                    
                    st.subheader("Batch Prediction Results")
                    st.dataframe(batch_df)
                    
                    # Provide download link
                    st.markdown(
                        get_table_download_link(batch_df, "churn_predictions.csv", "Download predictions as CSV"),
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")

        st.divider()

        # --- Feature Importance (Unchanged) ---
        st.header("🔍 Top Churn Drivers")
        with st.expander("View Model Feature Importances"):
            try:
                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)
                st.bar_chart(feat_imp_df.set_index('feature'), color="#00aaff")
            except Exception as e:
                st.error(f"Could not display feature importances: {e}")

        # --- Database Viewer (Unchanged) ---
        st.divider()
        st.header("🗄️ Past Prediction Log")
        if st.button("View All Past Predictions"):
            predictions_df = view_predictions()
            if predictions_df.empty:
                st.write("No predictions have been logged yet.")
            else:
                st.dataframe(predictions_df, use_container_width=True)

# --- RUN THE APP ---
if __name__ == "__main__":
    main()