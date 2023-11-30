#Importamos variables
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score

#Aqui las funciones:

def cambiar_texto(df, columna, antes, despues):
    if type(columna) == str or type(antes) == str or type(despues) == str:
        df[columna] = df[columna].str.replace(antes, despues)
        return df
    else:
        print("por favor, pasa los datos correctamente.")

def filtrar(df,columna_objetivo, texto, columna_nueva):
    df[columna_nueva] = df[columna_objetivo].apply(lambda x: 1 if texto in x else 0)
    return df

def primero(df, columna):

    generos = df[columna].str.split(';',expand=True)
    df[columna] = generos[0]
    return df



#Aqui cargamos la dataframe
df = pd.read_csv("../data/Raw/steam.csv")
df = df.dropna()
#Nota: Dropna solo ha quitado 15 filas, nada de lo que preocuparse.

#tenemos varias columnas que no nos aportan nada interesante, las quitamos.
df = df.drop(["appid", "owners"], axis=1)

#no utilizado al final, lo dejo incluido como comentario

"""publishers = df["publisher"].value_counts()

mascara = publishers[publishers < 2].index

# Eliminar las filas correspondientes a esos editores
df = df[~df['publisher'].isin(mascara)]"""

"""lista = list(df["steamspy_tags"].str.split(";"))
no_rep = []
for i in lista:
    for j in i:
           no_rep.append(j)"""

"""no_rep = pd.DataFrame(no_rep)
no_rep = list(dict(no_rep[0].value_counts().head(30)).keys())

no_rep"""

#se ve bien, vamos a empezar ese feature engineering.
#Nuestro target sera la columna de precio, por el momento, la voy a mantener, ya que si hago modficaciones a las filas, quiero al menos que se haga ahi tambien.

#voy a hacer varias DF, primero separare por categorias.
df["porcentage_positivas"] = round(df["positive_ratings"] / (df["positive_ratings"] + df["negative_ratings"]) *100, 2)
df=df[df["price"] < 50]
df=df[df["price"] != 0]
cat_df = df.copy()

#por el momento, quitare todo lo que no este relacionado con las categorias, tambien mantendremos las plataformas.
cat_df = cat_df.drop(["release_date", "positive_ratings","negative_ratings","developer","price"], axis=1)

label_encoder = LabelEncoder()
cat_df["publisher"] = label_encoder.fit_transform(cat_df["publisher"])

#vamos a dividirlos por los sistemas operativos que usan.

op_sys = cat_df['platforms'].str.split(';',expand=True)
#ahora les cambio el nombre, para tenerlas bien organizadas.
op_sys.columns = ["windows","mac","linux"]
#vamos a tener Nones, que se tratan como Nan
op_sys.fillna(0, inplace=True)

#op_sys = cambiar_texto(op_sys, "windows", "windows","1")
op_sys["windows"].value_counts()

#parece que se ha equivocado con los valores, vamos a arreglarlo.
#Linux parece no tener malos valores, como estaba mas a la derecha, vamos con mac directamente y metemos los valores que le falten a Linux
#Para no comerme posibles valores, primero haremos de mac a linux, y luego de windows al resto.

op_sys.loc[op_sys['mac'] == 'linux', 'linux'] = "linux"
op_sys.loc[op_sys['mac'] == 'linux'] = 0

#ahora windows
op_sys.loc[op_sys['windows'] == 'linux', 'linux'] = "linux"
op_sys.loc[op_sys['windows'] == 'mac', 'mac'] = "mac"
op_sys.loc[op_sys['windows'] != 'windows'] = 0

#ahora podemos cambiar el texto.
op_sys = cambiar_texto(op_sys, "windows", "windows","1")
op_sys = cambiar_texto(op_sys, "linux", "linux","1")
op_sys = cambiar_texto(op_sys, "mac", "mac","1")

#vamos a tener Nones, que se tratan como Nan
op_sys.fillna(0, inplace=True)

#Metemos las 3 nuevas que he creado.

cat_df = pd.concat([cat_df, op_sys], axis=1)

#vamos a dividirlos de forma basica, si es multijugador, single player, tiene Anti-cheat...

cat_df = filtrar(cat_df, 'categories', 'Multi-player', 'Multi-player')
cat_df = filtrar(cat_df, 'categories', 'Online', 'Online')
cat_df = filtrar(cat_df, 'categories', 'Single-player', 'Single-player')
cat_df = filtrar(cat_df, 'categories', 'Anti-Cheat', 'Anti-Cheat')

"""for i in no_rep:
    cat_df = filtrar(cat_df, 'steamspy_tags', i,i)"""

#para genre, vamos a ponerles solo el primero que tengan, luego lo convertiremos con label encoder

cat_df = primero(cat_df, "genres")

#cojemos las mas populares para crear columnas nuevas.

testeo = cat_df.copy()
testeo = filtrar(testeo, 'steamspy_tags', 'Indie', 'Indie')
testeo = filtrar(testeo, 'steamspy_tags', 'Action', 'Action')
testeo = filtrar(testeo, 'steamspy_tags', 'Adventure', 'Adventure')
testeo = filtrar(testeo, 'steamspy_tags', 'Casual', 'Casual')
testeo = filtrar(testeo, 'steamspy_tags', 'Strategy', 'Strategy')
testeo = filtrar(testeo, 'steamspy_tags', 'Simulation', 'Simulation')



test = pd.concat([testeo, df["price"]], axis=1)
test = test.drop(["name","english", "platforms", "categories", "genres", "steamspy_tags"],axis=1)

cat_df = pd.concat([cat_df,test["Indie"], test["Casual"], df["price"]],axis=1)
cat_df = cat_df.drop(["name", "platforms","categories", "genres", "steamspy_tags"],axis=1)
#por el momento, se quedara asi.

test = cat_df.tail(5000)
train = cat_df.drop(df.tail(5000).index)

#vamos a probar una regresion lineal, ya tenemos nuestra X e y
def regression(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)
    model = LinearRegression()

    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
    
    model.fit(X_train,y_train)

    pred = model.predict(test)

    print(X_train.shape)
    print(y_train.shape)
    print(pred.shape)
    print(test.shape)

    print("Media Target: ", y.mean())
    print("Intercept: ",model.intercept_)
    print("Model: ", model.coef_)
    print("R2", round(r2_score(obj_test, pred),3))
    print("MAE", round(mean_absolute_error(obj_test, pred),3))
    print("MAPE", round(mean_absolute_percentage_error(obj_test, pred),3))
    print("MSE", round(mean_squared_error(obj_test, pred),3))
    print("RMSE", round(np.sqrt(mean_squared_error(obj_test, pred)),3))
    filename = 'modelo_regression_lineal'

    with open(filename, 'wb') as archivo_salida:
        pickle.dump(model, archivo_salida)



#vamos a probar una regresion lineal, ya tenemos nuestra X e y
def polynominal(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)
    model = LinearRegression()

    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
    
    poly = PolynomialFeatures(degree=2)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scal = scaler.transform(X_train)
    X_test_scal = scaler.transform(test)
    poly.fit(X_train_scal)
    X_poly_train = poly.transform(X_train_scal)
    X_poly_test = poly.transform(X_test_scal)

    pol_reg4_reg = Ridge(alpha=1)
    pol_reg4_reg.fit(X_poly_train, y_train)

    pred = pol_reg4_reg.predict(X_poly_test)

    filename = 'modelo_regression_lineal'

    """with open(filename, 'wb') as archivo_salida:
        pickle.dump(model, archivo_salida)"""
    
#vamos a probar una regresion lineal, ya tenemos nuestra X e y
def reg_tree(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)

    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    pred = model.predict(test)

    filename = 'modelo_regression_lineal_cat.pickle'

    with open(filename, 'wb') as archivo_salida:
        pickle.dump(model, archivo_salida)

def guardar_csv(data,nombre):
    nombre = nombre +".csv"
    data.to_csv(f"../data/procesed/{nombre}", index=False)

def guardar_datos(data,nombre, direccion):
    nombre = nombre +".csv"
    data.to_csv(f"../data/{direccion}/{nombre}", index = False)

guardar_datos(train, "train_cat","train")
guardar_datos(test, "test_cat","test")
guardar_csv(cat_df, "categorias_df")