import os
import joblib
import streamlit as st
from sklearn import metrics
import time
import nltk
nltk.download('stopwords')
from pyresparser import ResumeParser


#  model path
model_path = os.path.join('model/', 'best_model.pkl')


# load model with joblib
model = joblib.load(model_path)

# prediction function. It returns the predicted result
def test(value):
   
    test_values = list()
    test_values2 = list()
    for i in value:
        test_values.append(i)
        
    test_values2.append(test_values)
    result = model.predict(test_values2)
    return result[0]
   
        
        
# function to check data type of values provided by the user
def checkDataType(data):
    if type(data)==str or type(data)==str:
        return str(data).title()
    if type(data)==list or type(data)==tuple:
        str_list=""
        for i,item in enumerate(data):
            str_list+=item+", "
        return str_list
    else:   return str(data)
    
    
    
# function for displaying prediction result
def predict(st, name, cv_path, age, gender, personality_values):
    
    
    try:
        predictions = test(personality_values)
    
        if gender == 0:
            gender = 'Male'
        else: 
            gender = 'Female'
    
        st.success('Name: {} '.format(name))
        st.success('Gender: {} '.format(gender))
        st.success('Age: {} '.format(age))
        
        data = ResumeParser(cv_path).get_extracted_data()
        del data['name']
        del data['mobile_number']
        
        for key in data.keys():
            if data[key] is not None:
                st.success("{}: {}".format(checkDataType(key.title()), checkDataType(data[key])))
                
        st.success('Predicted personality: {}'.format(predictions))
        
    except:
        st.error('CV is required')
    
            

def main():
    st.title('Personality Prediction')
    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Home':
        st.subheader('Fill in the form below')
        
        form = st.form(key='form')
        name = form.text_input("Fullname")
        gender = form.selectbox("Gender", ["Male", "Female"])
        age = form.number_input('Age', min_value=17, max_value=1000, help="17-26", step=1)
        openness = form.number_input('Openness Score', min_value=1, max_value=9,  help="1-9", step=1)
        neuroticism =  form.number_input('Neoroticism Score', min_value=1, max_value=9, help="1-9", step=1)
        conscientsciousness =  form.number_input('Consceintsciousness Score',  min_value=1, max_value=9, help="1-9", step=1)
        agreeableness=  form.number_input('Agreeableness Score',  min_value=1, max_value=9, help="1-9", step=1)
        extraversion =  form.number_input('Extraversion Score',  min_value=1, max_value=9, help="1-9", step=1)
        cv_path = form.file_uploader("Upload your CV", type=['pdf','docx'])
        submit_button = form.form_submit_button("Predict")
        
    
        if submit_button:
            if len(name) == 0:
                form.error('Name is required')
            else:
                if gender == 'Male':
                    gender = 0
                else: 
                    gender = 1
                    
                with st.spinner('Please wait...'):
                    time.sleep(5)
                
                form.write('Prediction Result')
                          
                predict(form, name, cv_path, age, gender, 
                        (gender, age, openness, neuroticism, 
                         conscientsciousness, agreeableness, extraversion)
                )           
        
    else:
        st.subheader('Brief details about each trait')
        
        st.write("""
                 * Openness: This trait encompasses characteristics such as insight, imagination, sensitivity, attentiveness, and curiosity. People who score high in openness are typically curious, creative, and open to new experiences.\n
                 
                 
                 * Conscientiousness: This trait relates to a person’s level of care, discipline, deliberation, and diligence. People who score high in conscientiousness are typically goal-oriented and have good self-control and organizational skills.\n
                 
                 
                 * Extroversion: This trait relates to a person’s level of emotional expression and assertiveness. Extroverted people are outgoing and comfortable interacting with others and tend to be enthusiastic and excitable.\n
                 
                 
                 * Agreeableness: This trait relates to a person’s level of generosity and cooperativeness. People who score high in agreeableness are typically kind, trusting, and considerate.\n
                 
                 
                 * Neuroticism: This trait relates to a person’s emotional stability and tendency to experience negative emotions such as anxiety and depression. People who score high in neuroticism are more easily prone to mood swings and may be more sensitive to stress.
                 
                 """)
        
        
        

if __name__ == '__main__':
    main()
    
    