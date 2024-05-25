import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

# Placeholder data for visualizations (replace with actual data from your training/testing set)
diabetes_accuracy = 0.85
heart_disease_accuracy = 0.87
diabetes_confusion_matrix = np.array([[50, 10], [5, 35]])  # Placeholder
heart_disease_confusion_matrix = np.array([[45, 12], [8, 30]])  # Placeholder

# Example dataframes for visualization
df_diabetes = pd.read_csv(f'{working_dir}/dataset/diabetes.csv')  # Placeholder path
df_heart = pd.read_csv(f'{working_dir}/dataset/heart.csv')  # Placeholder path

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    return fig

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level (mg/dL)')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value (mm Hg)')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value (mm)')

    with col2:
        Insulin = st.text_input('Insulin Level (mu U/mL)')

    with col3:
        BMI = st.text_input('BMI value (kg/m²)')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person (years)')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)


# Bar plot
    st.subheader('Bar plot')
    fig_barplot = plt.figure(figsize=(12, 10))
    for i, feature in enumerate(df_diabetes.columns[:-1]):
        avg_values = df_diabetes.groupby('Outcome')[feature].mean()
        plt.subplot(3, 3, i + 1)
        avg_values.plot(kind='bar', color=['blue', 'red'])
        plt.title(f'Average {feature} by Outcome')
        plt.xlabel('Outcome')
        plt.ylabel('Average ' + feature)
    plt.tight_layout()
    st.pyplot(fig_barplot)

    # Display confusion matrix
    st.subheader('Confusion Matrix')
    st.pyplot(plot_confusion_matrix(diabetes_confusion_matrix, 'Diabetes Prediction Confusion Matrix'))

    # Histograms
    st.subheader('Histograms')
    fig_hist = plt.figure(figsize=(12, 10))
    for i, feature in enumerate(df_diabetes.columns[:-1]):  # Assuming last column is the target
        plt.subplot(3, 3, i + 1)
        sns.histplot(data=df_diabetes, x=feature, kde=True)
        plt.title(f'{feature} Distribution')
        plt.xlabel('')
    plt.tight_layout()
    st.pyplot(fig_hist)

    

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age (years)')

    with col2:
        sex = st.text_input('Sex (1 = male; 0 = female)')

    with col3:
        cp = st.text_input('Chest Pain types (0-3)')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure (mm Hg)')

    with col2:
        chol = st.text_input('Serum Cholesterol (mg/dL)')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dL (1 = true; 0 = false)')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0-2)')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved (bpm)')

    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0-2)')

    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy (0-3)')

    with col1:
        thal = st.text_input('Thal (0 = normal; 1 = fixed defect; 2 = reversible defect)')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Bar plot
    st.subheader('Bar plot')
    fig_barplot = plt.figure(figsize=(12, 10))
    for i, feature in enumerate(df_heart.columns[:-1]):  # Assuming last column is the target
        avg_values = df_heart.groupby('target')[feature].mean()  # Replace 'Outcome' with 'target'
        plt.subplot(4, 4, i + 1)  # Adjust subplot layout if necessary
        avg_values.plot(kind='bar', color=['blue', 'red'])
        plt.title(f'Average {feature} by Outcome')
        plt.xlabel('Outcome')
        plt.ylabel('Average ' + feature)
    plt.tight_layout()
    st.pyplot(fig_barplot)

    # Display confusion matrix
    st.subheader('Confusion Matrix')
    st.pyplot(plot_confusion_matrix(heart_disease_confusion_matrix, 'Heart Disease Prediction Confusion Matrix'))

    # Histograms
    st.subheader('Histograms')
    fig_hist = plt.figure(figsize=(12, 10))
    for i, feature in enumerate(df_heart.columns[:-1]):  # Assuming last column is the target
        plt.subplot(4, 4, i + 1)  # Adjust subplot layout if necessary
        sns.histplot(data=df_heart, x=feature, kde=True)
        plt.title(f'{feature} Distribution')
        plt.xlabel('')
    plt.tight_layout()
    st.pyplot(fig_hist)

  
