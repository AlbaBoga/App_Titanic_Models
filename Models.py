import streamlit as st
from PIL import Image
#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#
# Tenemos dos opciones de layout, wide or center. Wide te lo adapta a la ventana
# mientras que center, lo centra.
st.set_page_config(page_title='TITANIC', page_icon='🚢', layout='centered')
st.set_option('deprecation.showPyplotGlobalUse', False)
#----------------------------CONFIGURACIÓN DE PÁGINAS----------------------------#

col1,col2,col3 = st.columns(3)
with col2:
    image2 = Image.open('img/logo1.webp')
    st.image(image2, width=250)

st.title('Análisis de los pasajeros del Titanic')

st.markdown("""

Se ha realizado un estudio de un conjunto de datos pertenecientes a registros de pasajeros dentro del Titanic.

En este dataset utilizado durante el análisis, se encuentran datos de gran relevancia relacionados con el hundimiento del Titanic. La creación de dicho barco fue de gran importancia en el siglo XX, ya que el trabajo de ingeniería realizado fue muy laborioso y suponía una conexión entre Inglaterra y Nueva York, siendo el barco de pasajeros más grande y lujoso de la historia. El Titanic albergó 2240 personas, entre tripulación y pasajeros, de los cuales murieron 1500. En este dataset, sólo hay constancia de 891 pasajeros, por lo que los diferentes descubrimientos van a ser en base a ellos, lo cuál supone 40% de las personas alojadas dentro del barco, perdiendo el 60% de la información restante. Para complementar la información dentro del dataset, se ha hecho uso de diferentes fuentes externas que han permitido llegar a conclusiones más acertadas en base a los datos obtenidos.

Procedimiento realizado:
- Primero se ha echado un primer vistazo a los datos.
- Seguidamente se ha realizado un trabajo de preprocesamiento de los datos donde se han buscado valores nulos, valores duplicados y, finalmente, se ha hecho una limpieza de las columnas pertinentes.
- Como tercer paso, se ha realizado una observación de los datos a través de diferentes gráficas, tablas y agrupaciones de datos.
- Depués, se ha implementado un modelo de clasificación para la estimación de la supervivencia de los pasajeros y un modelo de regresión para la estimación del precio de los billetes.
- Finalmente, se han resumido las conclusiones más importantes en base a esas observaciones.

Info: https://www.noaa.gov/gc-international-section/rms-titanic-history-and-significance#:~:text=The%20sinking%20of%20Titanic%20was,to%20improve%20safety%20of%20navigation%20.
        """
        )
#--------------------------------------TÍTULO-------------------------------------#



#--------------------------------------SIDEBAR-------------------------------------#
image1 = Image.open('img/logo.png')
st.sidebar.image(image1)

#--------------------------------------SIDEBAR-------------------------------------#