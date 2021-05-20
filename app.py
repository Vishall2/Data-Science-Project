import streamlit as st 
import pandas as pd
from pickle import dump
from pickle import load
from PIL import Image
from joblib import dump, load
import pybase64


# Page Heading...
col1, col2 = st.beta_columns(2)
col1.title('Fraud Analytics')

image = Image.open('excelr.png')
col2.image(image, caption=None, width=150, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

#st.title("Health Insurance Claim Fraud Detection")
st.subheader('@Author: Anitha-Jinson-Pranav-Santosh-Shrinivas-Vishal')

#col1, col2 = st.beta_columns(2)
#col1.title('Model Deployment: Random Forest')

#image = Image.open('excelr.png')
#col2.image(image, caption=None, width=350, use_column_width=200, clamp=False, channels='RGB', output_format='auto')


#image = Image.open('excelr.png')
#st.image(image, caption=None, width=100, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

#st.title('Model Deployment: Random Forest')
st.sidebar.header('User Input Parameters')

age_display = ("0 to 17","18 to 29","30 to 49","50 to 69","70 or Older")
options1 = list(range(len(age_display)))

surg_display = ("Medical","Surgical")
options2 = list(range(len(surg_display)))

emerg_display = ("Yes","No")
options3 = list(range(len(emerg_display)))

adm_display = ("Elective","Emergency","New Born","Not Available","Trauma","Urgent")
options4 = list(range(len(adm_display)))


def user_input_features():
    
    age = st.sidebar.selectbox("Age", options1, format_func=lambda x: age_display[x])
    days_spent = st.sidebar.number_input("Days Spent(1 to 120 days",min_value=1, max_value=120)
    admission_type = st.sidebar.selectbox('Admission Type',options4, format_func=lambda x: adm_display[x])
    home_self = st.sidebar.selectbox('Home/Self care',('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'))
    diagnosis_code = st.sidebar.number_input("Diagnosis Code")
    procedure_code = st.sidebar.number_input("Procedure Code")
    code_illness = st.sidebar.selectbox('Code Illness',('1','2','3','4'))
    mortality_risk = st.sidebar.selectbox('Mortality Risk',('1','2','3','4'))
    surg_description = st.sidebar.selectbox('Surgical Description',options2, format_func=lambda x: surg_display[x])
    emergency = st.sidebar.selectbox('Emergency Yes/No',options3, format_func=lambda x: emerg_display[x])
    #tot_cost = st.sidebar.number_input("Total Cost")
    #tot_charge = st.sidebar.number_input("Total Charge")
    ratio = st.sidebar.number_input("Ratio")
    payment_typology = st.sidebar.selectbox('Payment',('1','2','3','4'))
    
   
    data = {
            'AGE':age,
            'DAYS_SPENT':days_spent,
            'ADMISSION_TYPE':admission_type,
            'HOME_SELF CARE':home_self,
            'DIAGNOSIS CODE':diagnosis_code,
            'PROCEDURE CODE':procedure_code,
            'CODE ILLNESS':code_illness,
            'MORTALITY RISK':mortality_risk,
            'SURGICAL DESC':surg_description,
            'EMERGENCY':emergency,
            #'TOTAL COST':tot_cost,
            #'TOTAL CHARGE':tot_charge,
            'RATIO':ratio,
            'PAYMENT':payment_typology}
    
    features = pd.DataFrame(data,index = [0])
    return features 
    
df_depl = user_input_features()
st.subheader('User Input parameters')
st.write(df_depl)

# load the model from disk
#loaded_model = load(open('RF_Sample.sav', 'rb'))
#loaded_model = load(open('RF_Whole.sav', 'rb'))
#loaded_model = load(open('RF_Whole_no_std.sav', 'rb'))
#loaded_model = load(open('Logistic_Model.sav', 'rb'))
loaded_model = load(open('RF_WholeData_encoded.joblib', 'rb'))


prediction = loaded_model.predict(df_depl)
prediction_proba = loaded_model.predict_proba(df_depl)

st.subheader('Predicted Result')
st.write('Genuine' if prediction_proba[0][1] > 0.5 else 'Fraud')

st.subheader('Prediction Probability')
st.write(prediction_proba)


st.sidebar.title('About this project:')
st.sidebar.info('''The main aim of the project is to predict whether the Hospital 
             Insurance Claim is fraud or genuine.''')