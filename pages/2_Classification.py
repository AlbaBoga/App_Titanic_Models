#--------------LIBRERÍAS--------------#
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import scipy 
from scipy import stats
#--------------LIBRERÍAS--------------#

#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#
# Tenemos dos opciones de layout, wide or center. Wide te lo adapta a la ventana
# mientras que center, lo centra.
st.set_page_config(page_title='Estimador de supervivencia', page_icon='🧮', layout='centered')
st.set_option('deprecation.showPyplotGlobalUse', False)
#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#

#---------------------------------------------------------------COSAS QUE VAMOS A USAR EN TODA LA APP---------------------------------------------------------------#

titanic2 = pd.read_csv("data/titanic_limpio.csv")

#---------------------------------------------------------------COSAS QUE VAMOS A USAR EN TODA LA APP---------------------------------------------------------------#
st.title('Estimador de supervivencia de pasajeros')

titanic_class=titanic2[['Survived', 'Pclass', 'Sex',
       'Age', 'SibSp', 'Parch',
       'Level', 'Embarked']].copy()

# Codificador
encoder= OneHotEncoder(drop='first', sparse_output=False)
columnas=['Sex', 'Level', 'Embarked']
categorical_data = titanic_class[columnas]
encoded_categorical_data = pd.DataFrame(encoder.fit_transform(categorical_data))
encoded_categorical_data.columns = encoder.get_feature_names_out(columnas)
titanic_class = titanic_class.drop(['Sex', 'Level', 'Embarked'], axis=1)
encoded_data = pd.concat([titanic_class, encoded_categorical_data], axis=1)

encoded_data=encoded_data.astype('int64')

# Valores atípicos
z_scores = scipy.stats.zscore(encoded_data)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 4).all(axis=1) 
new_df = encoded_data[filtered_entries]

# test-train
X = new_df.drop(columns = ['Survived'], axis=1).copy() 
y = new_df[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=357)

# Modelo
model_GB_opt = GradientBoostingClassifier(learning_rate=0.5, max_depth=3,n_estimators=200, loss='deviance',
                                      subsample=1,min_samples_leaf=1,min_samples_split=4,random_state=357, validation_fraction=0.1,
                                        n_iter_no_change=5, tol=0.01)
model_GB_opt.fit(X = X_train, y = y_train)

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

def prediccion_survived(Pclass, Sex, Age, SibSp, Parch, Level, Embarked):
    data = pd.DataFrame(columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_Male', 'Level_B',
       'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T',
       'Embarked_Q', 'Embarked_S'])
    
    data.loc[0,'Pclass']=Pclass
    data.loc[0,'Age']=Age
    data.loc[0,'SibSp']=SibSp
    data.loc[0,'Parch']=Parch

    if Sex=='Male':
        data.loc[0,'Sex_Male']=1
    else:
        data.loc[0,'Sex_Male']=0

    if Embarked=='C':
        data.loc[0,['Embarked_Q','Embarked_S']]=0
    elif Embarked=='Q':
        data.loc[0, ['Embarked_Q', 'Embarked_S']] = [1, 0]
    else:
        data.loc[0, ['Embarked_Q', 'Embarked_S']] = [0, 1]

    if Level=='A':
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[0,0,0,0,0,0,0]
    elif Level=='B':
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[1,0,0,0,0,0,0]
    elif Level=='C':
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[0,1,0,0,0,0,0]
    elif Level=='D':
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[0,0,1,0,0,0,0]
    elif Level=='E':
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[0,0,0,1,0,0,0]
    elif Level=='F':
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[0,0,0,0,1,0,0]
    elif Level=='G':
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[0,0,0,0,0,1,0]
    else:
        data.loc[0,['Level_B',
                    'Level_C', 'Level_D', 'Level_E', 'Level_F', 'Level_G', 'Level_T']]=[0,0,0,0,0,0,1]
        
    predictions_GB = model_GB_opt.predict(data)
    
    return predictions_GB

prediction=prediccion_survived(Pclass, Sex, Age, SibSp, Parch, Level, Embarked)




if st.button('Supervivencia 👈'):
    if prediction==1:
        st.write('El pasajero ha sobrevivido al Titanic.')
    else:
        st.write('El pasajero pereció en el Titanic.')
else:
    st.write('📝 Estimando ... ')


if st.button('Volver 👈'):
    link = 'https://alba-app-titanic.streamlit.app/Clasificacion'
    st.markdown(f'<a href="{link}">Volver</a>', unsafe_allow_html=True)

#--------------------------------------SIDEBAR-------------------------------------#

image1 = Image.open('img/logo.png')
st.sidebar.image(image1)
#--------------------------------------SIDEBAR-------------------------------------#