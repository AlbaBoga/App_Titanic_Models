#--------------LIBRERÍAS--------------#
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pycaret.regression import *
#--------------LIBRERÍAS--------------#

#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#
# Tenemos dos opciones de layout, wide or center. Wide te lo adapta a la ventana
# mientras que center, lo centra.
st.set_page_config(page_title='Predictor de precios', page_icon='🧮', layout='centered')
st.set_option('deprecation.showPyplotGlobalUse', False)
#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#

#---------------------------------------------------------------COSAS QUE VAMOS A USAR EN TODA LA APP---------------------------------------------------------------#

titanic2 = pd.read_csv("data/titanic_limpio.csv")

#---------------------------------------------------------------COSAS QUE VAMOS A USAR EN TODA LA APP---------------------------------------------------------------#
st.title('Estimador para precio de billete')

st.subheader('Detalles del pasajero:')

col1,col2=st.columns(2)
with col1:
    Pclass = st.selectbox("Clase", titanic2['Pclass'].unique())
    Sex = st.selectbox("Género", titanic2['Sex'].unique())
    Age = st.number_input("Edad", min_value=0, max_value=80)
    SibSp = st.number_input("Número de hermanos o cónyuje a bordo", min_value=0, max_value=10)
with col2:
    Parch = st.number_input("Número de padres o hijos a bordo", min_value=0, max_value=10)
    Embarked = st.selectbox('Puerto de embarque',titanic2['Embarked'].unique())
    filtered_levels = titanic2[titanic2['Pclass'] == Pclass]['Level'].unique()
    Level = st.selectbox("Nivel del barco", filtered_levels)


def prediccion_fare(Pclass, Sex, Age, SibSp, Parch, Level, Embarked):
    data = pd.DataFrame({'Pclass': [Pclass],
                         'Sex': [Sex],
                         'Age': [Age],
                         'SibSp': [SibSp],
                         'Parch': [Parch],
                         'Level':[Level],
                         'Embarked': [Embarked]})
    
    loaded_model = load_model('titanic_reg')
    prediction = predict_model(loaded_model, data=data)
    
    return np.round(prediction.loc[0,'prediction_label'],2)

precio = prediccion_fare(Pclass, Sex, Age, SibSp, Parch, Level, Embarked)

if st.button('Precio 👈'):
    st.write('El precio estimado para el billete es de', precio, 'pounds')
else:
    st.write('📝 Estimando ... ')


if st.button('Volver 👈'):
    link = 'https://alba-app-titanic.streamlit.app/Regresion'
    st.markdown(f'<a href="{link}">Volver</a>', unsafe_allow_html=True)

#--------------------------------------SIDEBAR-------------------------------------#

image1 = Image.open('img/logo.png')
st.sidebar.image(image1)
#--------------------------------------SIDEBAR-------------------------------------#