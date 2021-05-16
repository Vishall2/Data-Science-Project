# Importing libraries..
import streamlit as st
import streamlit.components.v1 as components
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from sklearn.model_selection import StratifiedKFold 

from joblib import dump, load
import pybase64

import warnings
warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)

#model = load('model.joblib')


# page view setting...

st.set_page_config(layout='wide',initial_sidebar_state='expanded')
#st.markdown('<style>body{background-color: #c9eb34;}</style>',unsafe_allow_html=True)

#______________________________________________________________________________
# Page Heading...
st.title("Health Insurance Claim Fraud Detection")

st.subheader('@Author: Anitha Sriniwas Jinson Vishal Pranav Santosh')

    #from PIL import Image
    #image = Image.open('excelr.png')
    #st.image(image, caption=None, width=150, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Health Insurance Claim Fraud Detection ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    

# Setting up image to a streamlit webpage....

def main():
    
    col1, col2 = st.beta_columns(2)
    col1.title('Model Deployment: Random Forest')

    from PIL import Image
    image = Image.open('excelr.png')
    st.image(image, caption=None, width=150, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

#image = Image.open('excelr.png')
#st.image(image, caption=None, width=100, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

#st.title('Model Deployment: Random Forest')
#st.sidebar.title("User Options:")
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
#loaded_model = load(open('RF_WholeData_encoded.sav', 'rb'))
loaded_model = load(open('model.joblib','rb'))


prediction = loaded_model.predict(df_depl)
prediction_proba = loaded_model.predict_proba(df_depl)

st.subheader('Predicted Result')
st.write('Genuine' if prediction_proba[0][1] > 0.55 else 'Fraud')

st.subheader('Prediction Probability')
st.write(prediction_proba)

if __name__ == "__main__":
    main()

#
st.sidebar.title('About this project:')
st.sidebar.info('''The goal of this project is to build a model that can detect health insurance fraud..''')

