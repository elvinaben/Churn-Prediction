import streamlit as st
import joblib
import numpy as np

model = joblib.load('Xgboost_2.pkl')

def main():
    st.title('UTS Model Deployment - Churn Prediction 🏦')

    Unnamed: 0 = 0
    name = st.text_input("Name ")
    id = st.text_input("ID ", 1, 999999) # Data di soal => min:3, max:165033
    CustomerId = st.text_input("Customer ID", 10000000, 99999999) # cek lg! min:1.556570e+07, max:1.581569e+07
    CreditScore = st.number_input("Credit Score", 300, 1000) 
    
    Geography_Other = 0
    Geography_Germany = 0
    Geography_France = 0
    Geography_Spain = 0
    Geography_selection = st.radio("Country", ('Germany', 'France', 'Spain'))
    if Geography_selection == 'Germany':
        Geography_Germany = 1  
    elif Geography_selection == 'France':
        Geography_France = 1
    else:
        Geography_Spain = 1 

    Gender_selection = st.radio("👫 Gender", ('Male', 'Female'))
    if Gender_selection == 'Male':
        Gender = 1
    else:
        Gender = 0

    Age = st.number_input("Age", 17, 100)
    Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=1)
    Balance = st.number_input("💰 Balance", 0, 1000000)

    NumOfProducts = st.slider("Number Of Products", min_value=1, max_value=4, value=1)
    
    CC_selection = st.radio("💳 Credit Card", ('Yes', 'No'))
    if CC_selection == 'Yes':
        HasCrCard = 1
    else:
        HasCrCard = 0

    ActiveMember_selection = st.radio("Active Member?", ('Yes', 'No'))
    if ActiveMember_selection == 'Yes':
        IsActiveMember = 1
    else:
        IsActiveMember = 0 

    EstimatedSalary = st.number_input("💵 Estimated Salary", 0, 1000000)
    
    if st.button('Make Prediction'):
        features = [CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, 
                    HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, 
                    Geography_Germany, Geography_Spain, Geography_Other]
        result = make_prediction(features)
        st.success(f'{name} is predicted: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

# run di lokal
# streamlit run prediction.py
