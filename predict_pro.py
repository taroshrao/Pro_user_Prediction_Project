import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- MOVE THIS TO THE ABSOLUTE TOP ---
st.set_page_config(page_title="Pro User Prediction", layout="centered")
st.title("🧠 SkillCaptain Pro User Predictor")
st.markdown("Enter user details below to predict if they are likely to become a Pro User.")
# --- END OF MOVE ---

# --- 1. Load the Model and Scaler ---
# Make sure 'pro_user_nn_model.h5' and 'feature_scaler.pkl' are in the same directory as app.py
try:
    model = load_model('pro_user_nn_model.h5')
    scaler = joblib.load('feature_scaler.pkl')
    st.sidebar.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model or scaler: {e}")
    st.stop() # Stop the app if crucial files are missing

# --- 2. Define Expected Feature Columns (CRITICAL!) ---
# This list MUST match the columns in your X_train (before scaling), and in the same order.
# Get this list from your Colab notebook after X = df.drop('is_pro_user', axis=1)
# Example: X.columns.tolist()
# You will need to carefully copy this list from your Colab output.
# For simplicity, I'm providing a generic list that you'll need to replace with your actual columns.
# This includes the one-hot encoded columns in the correct order.
# You can run `print(X.columns.tolist())` in Colab after preprocessing to get this.
expected_features = [
    'has_seen_educator', 'consolidated_review_average', 'consolidated_review_param1',
    'consolidated_review_param2', 'consolidated_review_param3', 'consolidated_review_param4',
    'consolidated_review_param5', 'is_email_subscribed', 'total_continues_email_sent',
    'todo_mail_count', 'is_first_visit', 'num_goals_started', 'num_goals_completed',
    'num_assignments_submitted', 'num_assignments_reviewed', 'num_questions_asked',
    'days_since_last_goal_activity', 'num_orders_placed', 'num_orders_with_offer_code',
    'num_distinct_offer_codes_used', 'num_completed_orders', 'num_failed_orders',
    'days_since_registration', 'days_since_last_user_activity', 'goal_completion_rate',
    'user_type_WORKING',  # Example OHE column
    'dsa_ide_language_c++', 'dsa_ide_language_java', 'dsa_ide_language_javascript',
    'dsa_ide_language_python'  # Example OHE columns
]

# --- 3. Streamlit App Interface ---
#st.set_page_config(page_title="Pro User Prediction", layout="centered")
#st.title("🧠 SkillCaptain Pro User Predictor")
#st.markdown("Enter user details below to predict if they are likely to become a Pro User.")

st.sidebar.header("User Input Features")

# --- Define Input Widgets for Minimum Set of Features ---
# We'll pick a few key numerical and categorical features for user input
# For the rest, we'll use default values (e.g., 0 for counts, mean/median for others)

# Numerical Inputs
days_since_registration = st.sidebar.number_input("Days Since Registration", min_value=0, value=365)
days_since_last_user_activity = st.sidebar.number_input("Days Since Last Activity", min_value=0, value=30)
num_goals_completed = st.sidebar.number_input("Number of Goals Completed", min_value=0, value=0)
num_assignments_submitted = st.sidebar.number_input("Number of Assignments Submitted", min_value=0, value=0)
consolidated_review_average = st.sidebar.slider("Avg Review Score (0–5)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

# Categorical Inputs
user_type = st.sidebar.selectbox("User Type", ['STUDENT', 'WORKING'])
dsa_ide_language = st.sidebar.selectbox("DSA IDE Language", ['Unknown', 'python', 'java', 'c++', 'javascript'])

# --- 4. Preprocess User Input into a DataFrame ---
# Initialize a dictionary with default values for all features (mostly zeros)
# This ensures all 30 features are present and in the correct order
input_data = pd.DataFrame(0, index=[0], columns=expected_features)

# Populate with user inputs
input_data['days_since_registration'] = days_since_registration
input_data['days_since_last_user_activity'] = days_since_last_user_activity
input_data['num_goals_completed'] = num_goals_completed
input_data['num_assignments_submitted'] = num_assignments_submitted
input_data['consolidated_review_average'] = consolidated_review_average

# Handle one-hot encoded categorical features
if user_type == 'WORKING':
    input_data['user_type_WORKING'] = 1  # Assuming 'STUDENT' is the base (dropped) category

if dsa_ide_language != 'Unknown':
    # Ensure this matches your one-hot encoded column names exactly
    if f'dsa_ide_language_{dsa_ide_language}' in input_data.columns:
        input_data[f'dsa_ide_language_{dsa_ide_language}'] = 1
    else:
        st.warning(f"Warning: Column for DSA IDE Language '{dsa_ide_language}' not found in expected features. Please check `expected_features` list.")

# Fill other numerical features with means/medians if you want more realistic defaults
# For this basic example, we're assuming 0s are fine for most other count-based features
# Or, if you need more realistic defaults for other numerical features not explicitly taken as input:
# For example, to use the mean of training data for these:
# mean_values_X_train = pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=X.columns).mean()
# For col in ['total_continues_email_sent', 'todo_mail_count', ...]:
#     input_data[col] = mean_values_X_train[col]

# --- 5. Scale the Input Data ---
input_scaled = scaler.transform(input_data)

# --- 6. Make Prediction ---
if st.sidebar.button("Predict Pro User"):
    prediction_proba = model.predict(input_scaled)[0][0]  # Get the probability
    prediction_class = (prediction_proba > 0.5).astype(int)  # Binary 0/1 prediction

    st.subheader("Prediction Results:")
    st.write(f"**Probability of being a Pro User:** `{prediction_proba:.2f}`")

    if prediction_class == 1:
        st.success("This user is predicted to be a **PRO USER!** 🚀")
        st.balloons()
    else:
        st.info("This user is predicted to be a **NON-PRO USER.**")

    st.markdown("----")
    st.write("*Note: A higher probability indicates a stronger likelihood of being a Pro User. The prediction is based on a 0.5 threshold.*")
