import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from streamlit_option_menu import option_menu

# Replace 'path_to_dataset' with the actual path to your dataset file
data = pd.read_csv(r'C:\Users\User\Downloads\archive\Placement_Data_Full_Class.csv')

# loading the saved model
loaded_model = pickle.load(open('C:/Users/User/Machine Learning subject/MLProject/SVM WEB/trained_model.sav', 'rb'))

# function for prediction
def status_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not placed'
    else:
        return 'The person is placed'

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Campus Placement Prediction System',
                          
                          ['Status Prediction',
                           'Visual for Status by EmpTest',
                           'Visual for Status by Gender'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Status Prediction Page
if (selected == 'Status Prediction'):
    
    # page title
    st.title('Campus Placement by using SVM Model')
    
    

    gender = st.text_input('Gender (Male: 1, Female: 0)', key='gender')
    
    
    ssc_p = st.text_input('Secondary Education percentage - 10th Grade', key='ssc_p')
    
  
    ssc_b = st.text_input('Board of Education(10th Grade) : Central: 1, Others: 0 ', key='ssc_b')
    
   
    hsc_p = st.text_input('Higher Secondary Education percentage- 12th Grade', key='hsc_p')
 
    
    hsc_b = st.text_input('Board of Education(12th Grade) :Central: 1, Others: 0 ', key='hsc_b')

    
    hsc_s = st.text_input('Specialization in Higher Secondary Education : Commerse: 2, Science: 1, Arts: 0 (hsc_s)', key='hsc_s')


    degree_p = st.text_input('Degree Percentage ', key='degree_p')
    
    
    degree_t = st.text_input('Under Graduation(Degree type)- Field of degree education : Sci&tech: 1, Comm&Mgmt: 2, Others: 3 ', key='degree_t')
    
   
    workex = st.text_input('Work Experiance : Yes: 1, No: 0 ', key='workex')
    
    
    etest_p = st.text_input('Employability test percentage (conducted by college)', key='etest_p')
    
    
    specialisation = st.text_input('Post Graduation(MBA) - Specialization : Mkt&Fin: 1, Mkt&HR: 0 ', key='specialisation')
        

    mba_p = st.text_input('MBA Percentage', key='mba_p')

    # code for Prediction
    placement = ''
    
    # creating a button for Prediction
    
    if st.button('Placement Status Result'):
        placement = status_prediction([gender, ssc_p , ssc_b , hsc_p , hsc_b , hsc_s , degree_p , degree_t , workex , etest_p ,specialisation ,	mba_p])
        
        
        
    st.success(placement)

# Graph 1 Page
elif selected == 'Visual for Status by EmpTest':
    # page title
    st.title('Placement Status by Employment Test')

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x='etest_p', hue='status', fill=True)
    plt.xlabel('Employment Test')
    plt.ylabel('Density')
    plt.title('Density Plot of Status by Employment Test')
    st.pyplot(plt)

# Graph 2 page
elif selected == "Visual for Status by Gender":
    # page title
    st.title("Placement Status by Gender")

    plt.figure(figsize=(8, 6))
    gender_counts = data['gender'].value_counts()
    labels = ['Male', 'Female']
    colors = ['#8564ED', '#DD3DFE']
    plt.pie(gender_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Status by Gender')
    st.pyplot(plt)

    
