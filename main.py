import streamlit as st
import pandas as pd 
import pickle
import torch 
import time

with open('label_encoder.pkl', 'rb') as file:
    encoder_gender= pickle.load(file)
    
with open('one_hot_encoder.pkl', 'rb') as file:
    encoder_geo= pickle.load(file)
    
with open('processed_data_card.pkl', 'rb') as file:
    encoder_card= pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scalers=pickle.load(file)

loaded_model = torch.load("model.h5", weights_only=False)
loaded_model.eval()


# streamlit app
st.title('Customer Churn Predication')
st.sidebar.title("📊 Customer Churn Prediction")
st.subheader("👤 Customer Info")
# --- Personal Information ---
geography = st.selectbox('Geography', encoder_geo.categories_[0])
gender = st.selectbox('Gender', encoder_gender.classes_)
age = st.slider('Age', 18, 95, 30)

# --- Account & Banking Information ---
balance = st.number_input('Balance', min_value=0.0, step=100.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=100.0)
credit_card_score = st.number_input("Credit Card Score", min_value=0.0)

# --- Customer Profile ---
tenure = st.slider('Tenure (Years with Bank)', 0)
num_of_products = st.slider('Number Of Products', 1)
card_type = st.selectbox('Card Type', encoder_card.categories_[0])

# --- Engagement & Feedback ---
has_card = st.selectbox('Has Credit Card', ['False','True'])
has_card = 1 if has_card == 'True' else 0
is_active_mamber = st.selectbox('Is Active Member', ['False','True'])
is_active_member = 1 if is_active_mamber == 'True' else 0
complain = st.selectbox('Complain', ['False','True'])
complain = 1 if complain == 'True' else 0
satisfication_score = st.number_input('Satisfaction Score', min_value=1, max_value=5, step=1)
point_earned = st.number_input("Point Earned", min_value=0)


# Prepare input DataFrame
input_data = pd.DataFrame({
    "CreditScore": [credit_card_score],
    "Gender": [encoder_gender.transform([gender])[0]],   
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
    "Complain": [complain],
    "Satisfaction Score": [satisfication_score],
    "Point Earned": [point_earned]
})

# Encode geography
geo_encoded = encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(), 
    columns=encoder_geo.get_feature_names_out(["Geography"])
)
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Encode card type
card_encoded = encoder_card.transform([[card_type]])
card_encoded_df = pd.DataFrame(
    card_encoded.toarray(),
    columns=encoder_card.get_feature_names_out(["Card Type"])
)
input_data = pd.concat([input_data, card_encoded_df], axis=1)

# Scale data
input_data_scaled = scalers.transform(input_data)

# Prediction
prediction = loaded_model(torch.tensor(input_data_scaled,dtype=torch.float32))

churn_probability = torch.sigmoid(prediction).item()

with st.spinner("Analyzing..."):
    time.sleep(1)  # simulate processing
    # st.write("### 🔮 Prediction Result")
    st.write(f"Churn Probability: **{churn_probability:.2f}**")

    # ---- Personalization ----
    if churn_probability > 0.7:
        st.error("⚠️ Customer likely to churn. Suggest **retention offers** (discounts, loyalty points).")
    elif churn_probability < 0.3:
        st.success("✅ Customer is loyal! Keep them engaged with **exclusive content & rewards**.")
    else:
        st.warning("🤔 Customer is uncertain. Monitor behavior & send **targeted offers**.")