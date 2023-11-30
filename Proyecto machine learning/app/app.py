import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as c
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import seaborn as sns
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="Proyecto Machine learning",
                   page_icon=":thumbs_up:")
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
        <style>
        body {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return  

st.markdown(
    """
    <style>
        .sidebar-content .stSelectbox label span {{
            color: white;  
            text-shadow: -1px -1px 1px black, 1px -1px 1px black, -1px 1px 1px black, 1px 1px 1px black;  
        }}
    </style>
    """,
    unsafe_allow_html=True
)

bg = '../docs/panda.png'
sidebar_bg(bg)
set_png_as_page_bg(bg)

seleccion = st.sidebar.selectbox("Selecciona menu", ['Dataframe juegos','Dataframe Reviews'])

if seleccion == "Dataframe juegos":
    st.title("Dataframe de juegos") 
    st.write("Idea original. Aqui analizaremos la dataframe de juegos de steam, sacada con steamspy API.")
    
    # Establecer una imagen como fondo con CSS


    with st.expander("Introducción"):
        st.write("Intentar predecir el precio de un videojuego. \n El objetivo es poder ayudar a los devs a poder realizar estudios rapidamente y poder ahorrar dinero y recursos.\n",
                 "\nVamos a empezar analizando la dataframe y viendo que valores y variables podemos aprovechar. \n")

    with st.expander("Raw Dataframe"):
        df = pd.read_csv("../data/Raw/steam.csv", sep=",")
        st.write("Asi era la Dataframe antes de empezar.")
        st.write(df.head())
        st.write("Resumen de las columnas:")
        st.write("appid: ID segun la API.\n",
        "\nname: Nombre del juego.\n",
        "\nrelease_date: Fecha de lanzamiento.\n",
        "\nenglish: Si tiene traducion al ingles (1) o no (0).\n",
        "\ndeveloper: Quien ha creado el juego.\n",
        "\npublisher: quien publica el juego.\n",
        "\nPlatform: SO donde se puede jugar el juego.\n",
        "\nRequired age: edad minima recomendada.\n",
        "\ncategories: Categorias a las que pertenece el juego.\n",
        "\ngenres: Genero al que pertenece el juego.\n",
        "\nsteamspy_tags: etiquetas para catalogar el juego.\n",
        "\nachievements: Cuantos desafios contiene.\n",
        "\npositive_ratings: Numero de reviews positivas.\n",
        "\nnegative_ratings: Numero de reviews negativas.\n",
        "\naverage_playtime: tiempo medio de juego entre jugadores.\n",
        "\nmedian_playtime: media de tiempo de juego.\n",
        "\nowners: Cantidad de usuarios que han realizado la compra.\n",
        "\nprice: precio del juego, nuestra variable tarjet.\n")
        st.write("Dataframe Filtrada.")
        filtro = st.sidebar.selectbox("Selecciona que columna deseas ver.", df.columns)
        df_filtered = df[filtro]
        st.write(df_filtered.head())
        
    with st.expander("Dataframe limpiada"):
        df = pd.read_csv("../data/procesed/categorias_df.csv", sep=",")
        
        st.write("Dataframe limpiada y lista para usarse en un modelo.")
        st.write(df.head())
        st.write("Las nuevas variables son principalmente valores entre 0 y 1 para indicar si tiene dicha caracteristica.")



    with st.expander("prueba de prediccion"):
        train = pd.read_csv("../data/train/train_cat.csv", sep=",")
        test = pd.read_csv("../data/test/test_cat.csv", sep=",")
        with open("../models/modelo_regression_lineal_cat.pkl", "rb") as modelo:
            model = pickle.load(modelo)
        if st.button("Realizar prediccion:"):
            target = "price"
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])
            obj = train[target]
            obj_test = test[target]
            test = test.drop([target],axis=1)
            train = train.drop([target],axis=1)


            X = train
            y = obj
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
            model.fit(X_train,y_train)

            pred = model.predict(test)
            st.success(f"Los resultados de la prediccion son:\n Margen de error de {round(mean_absolute_error(obj_test, pred),3)} \n porcentaje de fallo de {round(mean_absolute_percentage_error(obj_test, pred)*100,3)}%")

    with st.expander("Porque no funciona."):
        st.write("Desgraciadamente, y segun podemos ver en el mapa de calor mas abajo, no hay suficientes correlaciones para que el modelo aprenda.")
        imagen = Image.open("../docs/cat_heatmap.png")
        st.image(imagen, caption='Mapa de calor.', use_column_width=True)
        st.write("Esto es debido, a que el precio es demasiado bajo en general, el modelo no consigue aprender del todo bien.")
        imagen = Image.open("../docs/cat_price_hist.png")
        st.image(imagen, caption='Historico del precio.', use_column_width=True)



elif seleccion == "Dataframe Reviews":

    st.title("Dataframe de Reviews")
    st.write("Aqui vamos a analizar juegos para intentar predecir la User Score en la pagina de Metacritic")

    with st.expander("Introducción"):
        st.write("Predecir las reviews que obtienen los juegos segun sus características.")

    with st.expander("Tabla"):

        df = pd.read_csv("../data/Raw/metacritic_games.csv", sep=",")
        st.write("Dataframe Pura, sin manipular")
        st.write(df.head())
        st.write("Resumen de las columnas:")
        st.write("\ngame: Nombre del juego.\n",
        "\nplatform: plataforma donde se puede jugar el juego.\n",
        "\ndeveloper: Quien ha creado el juego.\n",
        "\ngenre: Genero al que pertenece el juego.\n",
        "\nnumber_players: Si tiene o no mutijugador online\n",
        "\nrating: PEGI basicamente.\n",
        "\nrelease_date: Fecha de lanzamiento.\n",
        "\npositive_critics: Numero de reviews positivas.\n",
        "\nneutral_critics: criticas ni buenas ni malas.\n",
        "\nnegative_critics: Numero de reviews negativas.\n",
        "\npositive_users: Numero de usuarios contentos\n",
        "\nneutral_users: numero de usuarios ni contentos ni disgustados.\n",
        "\nnegative_users: numero de usuarios disgustados.\n",
        "\nmetascore: Score dado por la propia metacritic.\n",
        "\nuser_score: Media de todas las scores de los usuarios.\n")

        st.write("Dataframe filtrada.")
        filtro = st.sidebar.selectbox("Selecciona que columna deseas ver.", df.columns)
        df_filtered = df[filtro]
        st.write(df_filtered.head())
        
    with st.expander("Tabla limpiada"):
        df = pd.read_csv("../data/procesed/reviews.csv", sep=",")
        
        st.write("Dataframe limpiada y lista para usarse en un modelo.")
        st.write(df.head())
        st.write("Las nuevas variables son principalmente valores entre 0 y 1 para indicar si tiene dicha caracteristica.")



    with st.expander("prueba de prediccion"):
        train = pd.read_csv("../data/train/train.csv", sep=",")
        test = pd.read_csv("../data/test/test.csv", sep=",")
        st.write("Por favor, selecciona el metodo de prediccion que deseas.")
        st.write("Facilitamos los modelos utilizados, sin embargo, recomendamos el modelo de Arbol Aleatorio.")
        if st.button("Modelo Lineal:"):
            with open("../models/modelo_regression_lineal.pkl", "rb") as modelo:
                linear_model = pickle.load(modelo)
            target = "user_score"
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])
            obj = train[target]
            obj_test = test[target]
            test = test.drop([target],axis=1)
            train = train.drop([target],axis=1)
            #st.write(target, obj, obj_test, train, test)

            X = train
            y = obj
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
            linear_model.fit(X_train,y_train)

            pred1 = linear_model.predict(test)
            st.write("Modelo que utiliza una prediccion lineal, resultados aceptables.")
            st.success(f"Los resultados de la prediccion son:\n Margen de error de {round(mean_absolute_error(obj_test, pred1),3)} \n porcentaje de fallo de {round(mean_absolute_percentage_error(obj_test, pred1)*100,3)}%")

        if st.button("Modelo polynominal:"):
            with open("../models/modelo_regression_polynominal.pkl", "rb") as modelo:
                poly_model = pickle.load(modelo)
            target = "user_score"
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])
            obj = train[target]
            obj_test = test[target]
            test = test.drop([target],axis=1)
            train = train.drop([target],axis=1)


            X = train
            y = obj
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
            poly_model.fit(X_train,y_train)

            pred2 = poly_model.predict(test)
            st.write("Modelo que combina el lineal con escalados, ligeramente mejor.")
            st.success(f"Los resultados de la prediccion son:\n Margen de error de {round(mean_absolute_error(obj_test, pred2),3)} \n porcentaje de fallo de {round(mean_absolute_percentage_error(obj_test, pred2)*100,3)}%")

        if st.button("Modelo Arbol aleatorio:"):
            with open("../models/modelo_regression_RFR.pkl", "rb") as modelo:
                rfr_model = pickle.load(modelo)
            target = "user_score"
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])
            obj = train[target]
            obj_test = test[target]
            test = test.drop([target],axis=1)
            train = train.drop([target],axis=1)

            X = train
            y = obj
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

            #rfr_model.fit(X_train,y_train)

            pred3 = rfr_model.predict(test)
            st.write("Modelo de estilo arbol, que toma decisiones segun las condiciones, resultados muy precisos.")
            st.success(f"Los resultados de la prediccion son:\n Margen de error de {round(mean_absolute_error(obj_test, pred3),3)} \n porcentaje de fallo de {round(mean_absolute_percentage_error(obj_test, pred3)*100,3)}%")

        if st.button("Modelo conexion vecinos:"):
            with open("../models/modelo_regression_KNR.pkl", "rb") as modelo:
                knr_model = pickle.load(modelo)
            target = "user_score"
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])
            obj = train[target]
            obj_test = test[target]
            test = test.drop([target],axis=1)
            train = train.drop([target],axis=1)

            X = train
            y = obj
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

            knr_model.fit(X_train,y_train)

            pred4 = knr_model.predict(test)
            st.write("Modelo que agrupa segun los resultados mas cercanos, buena precisión.")
            st.success(f"Los resultados de la prediccion son:\n Margen de error de {round(mean_absolute_error(obj_test, pred4),3)} \n porcentaje de fallo de {round(mean_absolute_percentage_error(obj_test, pred4)*100,3)}%")


        if st.button("Modelo gradiente:"):
            with open("../models/modelo_regression_GBR.pkl", "rb") as modelo:
                gbr_model = pickle.load(modelo)
            target = "user_score"
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])
            obj = train[target]
            obj_test = test[target]
            test = test.drop([target],axis=1)
            train = train.drop([target],axis=1)

            X = train
            y = obj
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

            gbr_model.fit(X_train,y_train)

            pred5 = gbr_model.predict(test)
            st.write("Modelo basado en arboles, como el anterior, pero esta vez, estan hechos de forma secuencial, desgraciadamente da el peor resultado.")
            st.success(f"Los resultados de la prediccion son:\n Margen de error de {round(mean_absolute_error(obj_test, pred5),3)} \n porcentaje de fallo de {round(mean_absolute_percentage_error(obj_test, pred5)*100,3)}%")

        if st.button("Modelo Vectorial:"):
            with open("../models/modelo_regression_SVR.pkl", "rb") as modelo:
                svr_model = pickle.load(modelo)
            target = "user_score"
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])
            obj = train[target]
            obj_test = test[target]
            test = test.drop([target],axis=1)
            train = train.drop([target],axis=1)

            X = train
            y = obj
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

            svr_model.fit(X_train,y_train)

            pred6 = svr_model.predict(test)
            st.write("Modelo lineal basado en soportes vectoriales.")
            st.success(f"Los resultados de la prediccion son:\n Margen de error de {round(mean_absolute_error(obj_test, pred6),3)} \n porcentaje de fallo de {round(mean_absolute_percentage_error(obj_test, pred6)*100,3)}%")

    with st.expander("Descripcion de nuestro modelo final."):
        train = pd.read_csv("../data/train/train.csv", sep=",")
        test = pd.read_csv("../data/test/test.csv", sep=",")
        
        st.write("En esta seccion, vamos a profundizar un poco en nuestra dataframe y modelo.")
        with open("../models/modelo_regression_RFR.pkl", "rb") as modelo:
            rfr_model = pickle.load(modelo)
        target = "user_score"
        train = train.select_dtypes(exclude=['object'])
        test = test.select_dtypes(exclude=['object'])
        obj = train[target]
        obj_test = test[target]
        test = test.drop([target],axis=1)
        train = train.drop([target],axis=1)

        X = train
        y = obj
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

        #rfr_model.fit(X_train,y_train)
        pred3 = rfr_model.predict(test)


        imagen = Image.open("../docs/heatmap_rev.png")
        st.write("Segun vemos en el mapa de calor, tenemos mas correlaciones en nuestra variable objetivo, que es la user_score.")
        st.image(imagen, caption='Mapa de calor.', use_column_width=True)
        st.write("Aqui debajo tenemos el historico.")
        imagen = Image.open("../docs/historico_rev.png")
        st.image(imagen, caption='Historico de user_score', use_column_width=True)
        st.write("EL historico contiene la cantidad de scores y su nota.")
        mapa = pd.DataFrame({"predicciones":pred3, "variable objetivo":obj_test})

        st.write("Vamos a hacer una comparación, primero veamos un scatterplot del modelo de Regression Lineal.")
        imagen = Image.open("../docs/comparativa.png")
        st.image(imagen, caption='Historico de user_score', use_column_width=True)
        st.write("Debajo tenemos otras 2 graficas, ambas son un scatterplot en el que podemos ver la similitud del modelo con la variable objetivo.")
        

        fig, ax = plt.subplots()
        sns.scatterplot(data=mapa["predicciones"], color="blue")
        #ax.legend(title="variable", bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_xlabel('index')
        ax.set_ylabel('Valor variable')
        ax.set_title('Scatter predicciones')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(data=mapa["variable objetivo"], color="red")
        #ax.legend(title="variable", bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_xlabel('index')
        ax.set_ylabel('Valor variable')
        ax.set_title('Scatter variable objetivo')
        st.pyplot(fig)



        st.write("vamos a verlo ahora con un Box plot.")


        fig, ax = plt.subplots()
        sns.boxplot(data=mapa["predicciones"].reset_index(drop=True))
        #ax.legend(title="variable", bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_ylabel('Valor variable')
        ax.set_title('Boxplot predicciones')
        st.pyplot(fig)




        fig, ax = plt.subplots()
        sns.boxplot(data=mapa["predicciones"].reset_index(drop=True))
        #ax.legend(title="variable", bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_ylabel('Valor variable')
        ax.set_title('Boxplot variable objetivo')
        st.pyplot(fig)

        st.write("Los boxplot nos muestran que los valores medios estan entre 63-78 y los valores no comunes son inferiores a 43.",
                 "\nEsto es normal, hay muchos juegos que tienen notas mas bajas, pero por norma general, suele estar la puntuacion entre 70-80\n")
