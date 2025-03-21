import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", 'rb'))

# Mapping for categorical values
location_mapping = {'Rural': 0, 'Urban': 1}
cell_phone_mapping = {'No': 0, 'Yes': 1}
gender_mapping = {'Female': 0, 'Male': 1}
relationship_with_head_mapping = {
    'Spouse': 0, 'Head of Household': 1, 'Other relative': 2, 
    'Child': 3, 'Parent': 4, 'Other non-relatives': 5
}
marital_status_mapping = {
    'Married/Living together': 0, 'Widowed': 1, 'Single/Never Married': 2,  
    'Divorced/Separated': 3, 'Dont know': 4
}
education_level_mapping = {
    'Secondary education': 0, 'No formal education': 1,
    'Vocational/Specialised training': 2, 'Primary education': 3,
    'Tertiary education': 4, 'Other/Dont know/RTA': 5
}
job_type_mapping = {
    'Self employed': 0, 'Government Dependent': 1, 'Formally employed Private': 2, 
    'Informally employed': 3, 'Formally employed Government': 4, 
    'Farming and Fishing': 5, 'Remittance Dependent': 6, 
    'Other Income': 7, 'Dont Know/Refuse to answer': 8, 'No Income': 9
}

# Function for bank account prediction
def bank_prediction(input_data):
    input_data_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_array)
    
    if prediction[0] == 'No':
        return "Does not have a bank account"
    else:
        return "Has a bank account"

# Streamlit UI
def main():
    st.title("Bank Account Prediction Web App")
    st.header("Enter Details")

    Age = st.number_input('Age')
    Household_size = st.number_input('Enter household size (number of people):')
    Location = st.selectbox('Select location type (Urban/Rural)', list(location_mapping.keys()))
    Cell_Phone = st.selectbox('Do you have cellphone access?', list(cell_phone_mapping.keys()))
    Gender = st.selectbox('Gender?', list(gender_mapping.keys()))
    relationship_with_head = st.selectbox('Select relationship with head:', list(relationship_with_head_mapping.keys()))
    Marital_status = st.selectbox('Select Marital Status?', list(marital_status_mapping.keys()))
    Education_level = st.selectbox('Select Education Level', list(education_level_mapping.keys()))
    job_type = st.selectbox('Select Job Type', list(job_type_mapping.keys()))

    predictt = ""
    if st.button("Predict"):
        try:
            input_data = {
                f'location_type_{Location}': 1,
                f'cellphone_access_{Cell_Phone}': 1,
                f'gender_of_respondent_{Gender}': 1,
                f'relationship_with_head_{relationship_with_head}': 1,
                f'marital_status_{Marital_status}': 1,
                f'education_level_{Education_level}': 1,
                f'job_type_{job_type}': 1,
                'household_size': Household_size,
                'age_of_respondent': Age
            }

            input_df = pd.DataFrame([input_data])

            # Ensure expected columns exist
            expected_columns = ['location_type_Urban', 'cellphone_access_Yes',
                'gender_of_respondent_Male', 'relationship_with_head_Head of Household',
                'relationship_with_head_Other non-relatives', 'relationship_with_head_Other relative',
                'relationship_with_head_Parent', 'relationship_with_head_Spouse',
                'marital_status_Dont know', 'marital_status_Married/Living together',
                'marital_status_Single/Never Married', 'marital_status_Widowed',
                'education_level_Other/Dont know/RTA', 'education_level_Primary education',
                'education_level_Secondary education', 'education_level_Tertiary education',
                'education_level_Vocational/Specialised training', 'job_type_Farming and Fishing',
                'job_type_Formally employed Government', 'job_type_Formally employed Private',
                'job_type_Government Dependent', 'job_type_Informally employed', 
                'job_type_No Income', 'job_type_Other Income', 'job_type_Remittance Dependent',
                'job_type_Self employed', 'household_size', 'age_of_respondent'
            ]

            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[expected_columns]

            predictt = bank_prediction(input_df)
            st.success(predictt)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Ensure this is at the correct indentation level (no spaces before)
if __name__ == "__main__":
    main()
