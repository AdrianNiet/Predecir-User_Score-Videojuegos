#importamos librerias

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score

#para guardar los datos.

def guardar_datos(data,nombre, direccion):
    nombre = nombre +".csv"
    data.to_csv(f"../data/{direccion}/{nombre}",index=False)


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


    filename = 'modelo_regression_lineal.pkl'



    fig, ax = plt.subplots()
    sns.scatterplot(pred, label = "predicciones.")
    sns.scatterplot(obj_test.reset_index(drop=True), label= "Objetivo")
    ax.legend(title="variable", bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_title('Scatter pred VS obj')
    plt.show()

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

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scal = scaler.transform(X_train)
    X_test_scal = scaler.transform(test)
    poly.fit(X_train_scal)
    X_poly_train = poly.transform(X_train_scal)
    X_poly_test = poly.transform(X_test_scal)

    pol_reg4_reg = Ridge(alpha=0.1)
    pol_reg4_reg.fit(X_poly_train, y_train)

    pred = pol_reg4_reg.predict(X_poly_test)
    filename = 'modelo_regression_polynominal.pkl'

    with open(filename, 'wb') as archivo_salida:
        pickle.dump(model, archivo_salida)

def grid_rfr(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)


    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
    
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor())
    ])

    parameters = {
        'scaler__with_mean': [True, False],
        'scaler__with_std': [True, False],
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 5, 10],
        'regressor__min_samples_split': [2, 5, 10]
    }

    rfr_gs = GridSearchCV(estimator=pipe, param_grid=parameters, cv=3, scoring="accuracy", verbose=3, n_jobs=-1)
    rfr_gs.fit(X_train, y_train)
    final_rfr = rfr_gs.best_estimator_
    final_rfr.fit(X_train, y_train)
    pred = final_rfr.predict(test)

    sns.scatterplot(pred)
    sns.scatterplot(obj_test.reset_index(drop=True))
    filename = 'modelo_regression_RFR.pkl'

    with open(filename, 'wb') as archivo_salida:
        pickle.dump(rfr_gs, archivo_salida)


def grid_gbr(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)


    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
    
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor())
    ])

    parameters = {
            'scaler__with_mean': [True, False],
            'scaler__with_std': [True, False],
            "regressor__n_estimators":[50,100,150],
            "regressor__max_depth": [2,3,4,5],
            "regressor__max_features": [2,3,4],
            "regressor__learning_rate":[0.01,0.1,0.5]
            }

    gbr_gs = GridSearchCV(estimator=pipe, param_grid=parameters, cv=3, scoring="accuracy", verbose=3, n_jobs=-1)
    gbr_gs.fit(X_train, y_train)
    final = gbr_gs.best_estimator_
    final.fit(X_train, y_train)
    pred = final.predict(test)


    print("Media Target: ", y.mean())
    print("R2", round(r2_score(obj_test, pred),3))
    print("MAE", round(mean_absolute_error(obj_test, pred),3))
    print("MAPE", round(mean_absolute_percentage_error(obj_test, pred),3))
    print("MSE", round(mean_squared_error(obj_test, pred),3))
    print("RMSE", round(np.sqrt(mean_squared_error(obj_test, pred)),3))

    filename = 'modelo_regression_GBR.pkl'

    with open(filename, 'wb') as archivo_salida:
        pickle.dump(final, archivo_salida)

def grid_knn(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)


    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
    
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ("selectkbest", SelectKBest()),
    ('regressor', KNeighborsRegressor())
    ])

    parameters = {
        'scaler__with_mean': [True, False],
        'scaler__with_std': [True, False],
        'selectkbest__k':np.arange(5,15),
        'regressor__n_neighbors': np.arange(5,15)
        }

    knr_gs = GridSearchCV(estimator=pipe, param_grid=parameters, cv=3, scoring="accuracy", verbose=3, n_jobs=-1)
    knr_gs.fit(X_train, y_train)
    final = knr_gs.best_estimator_
    final.fit(X_train, y_train)
    pred = final.predict(test)


    print("Media Target: ", y.mean())
    print("R2", round(r2_score(obj_test, pred),3))
    print("MAE", round(mean_absolute_error(obj_test, pred),3))
    print("MAPE", round(mean_absolute_percentage_error(obj_test, pred),3))
    print("MSE", round(mean_squared_error(obj_test, pred),3))
    print("RMSE", round(np.sqrt(mean_squared_error(obj_test, pred)),3))

    filename = 'modelo_regression_KNR.pkl'

    with open(filename, 'wb') as archivo_salida:
        pickle.dump(final, archivo_salida)


def grid_SVR(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)


    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)
    
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ("selectkbest", SelectKBest()),
    ('regressor', SVR())
    ])

    parameters = {
        'scaler__with_mean': [True, False],
        'scaler__with_std': [True, False],
        'selectkbest__k':np.arange(5,15),
        'regressor__C': [0.1,1,10]
        }
    
    svr_gs = GridSearchCV(estimator=pipe, param_grid=parameters, cv=3, scoring="accuracy", verbose=3, n_jobs=-1)
    svr_gs.fit(X_train, y_train)
    final = svr_gs.best_estimator_
    final.fit(X_train, y_train)
    pred = final.predict(test)


    print("Media Target: ", y.mean())
    print("R2", round(r2_score(obj_test, pred),3))
    print("MAE", round(mean_absolute_error(obj_test, pred),3))
    print("MAPE", round(mean_absolute_percentage_error(obj_test, pred),3))
    print("MSE", round(mean_squared_error(obj_test, pred),3))
    print("RMSE", round(np.sqrt(mean_squared_error(obj_test, pred)),3))

    filename = 'modelo_regression_SVR.pkl'

    with open(filename, 'wb') as archivo_salida:
        pickle.dump(final, archivo_salida)

def kmeans(train,test, target):
    train = train.select_dtypes(exclude=['object'])
    test = test.select_dtypes(exclude=['object'])
    obj = train[target]
    obj_test = test[target]
    test = test.drop([target],axis=1)
    train = train.drop([target],axis=1)

    kmeans = KMeans()
        
    X = train
    y = obj

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1)

    # Definir el modelo de PCA
    pca = PCA()

    # Definir el modelo de KMeans
    kmeans = KMeans(n_init=10)

    # Definir la tubería con PCA seguido de KMeans
    pipe = Pipeline([
        ('pca', pca),
        ('kmeans', kmeans)
    ])

    # Definir el espacio de búsqueda de hiperparámetros
    param_grid = {
        'pca__n_components': [2, 5, 10],  # Ajusta según la cantidad deseada de componentes principales
        'kmeans__n_clusters': [5, 10, 15, 20]  # Ajusta los valores según sea necesario
    }

    # Configurar Grid Search con validación cruzada
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='adjusted_rand_score')  # Ajusta la métrica según tu problema

    # Entrenar el modelo con el conjunto de entrenamiento
    grid_search.fit(X_train)

    # Obtener el mejor modelo y sus hiperparámetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    predictions = best_model.predict(X_test)

    print("Mejores hiperparámetros:", best_params)
    print("Predicciones en el conjunto de prueba:", predictions)
    print("Resultados de la validación cruzada:")
    print(grid_search.cv_results_)

    inercia = best_model.named_steps['kmeans'].inertia_
    print("Inercia del modelo de KMeans:", inercia)




#abrimos la databse

df = pd.read_csv("../data/Raw/metacritic_games.csv")
test = df.copy()

#Empezamos a manipularla, primero quitamos los NAN.

df["number_players"] = df["number_players"].fillna("no se sabe")
df["rating"] = df["rating"].fillna("nada")
df["genre"] = df["genre"].fillna("otro")
df["developer"] = df["developer"].fillna("Desconocido")

#vamos a ir creando columnas, vamos a sacar la media de scores segun el genero.

test = df.groupby("genre")["user_score"].mean().reset_index()
test.columns = ["genre","media"]
test2 = df.groupby("genre")["metascore"].mean().reset_index()
test2.columns = ["genre","media"]

df['Media_usuario'] = np.nan
df['Media_meta'] = np.nan

for _, row in test.iterrows():
    categoria = row['genre']
    media_correspondiente = row['media']
    df.loc[df['genre'] == categoria, 'Media_usuario'] = media_correspondiente

for _, row in test2.iterrows():
    categoria = row['genre']
    media_correspondiente = row['media']
    df.loc[df['genre'] == categoria, 'Media_meta'] = media_correspondiente

#convertimos a codigo numerico las variables de texto.
label_encoder = LabelEncoder()
df["platform"] = label_encoder.fit_transform(df["platform"])
df["genre"] = label_encoder.fit_transform(df["genre"])
df["number_players"] = label_encoder.fit_transform(df["number_players"])
df["rating"] = label_encoder.fit_transform(df["rating"])

#quitamos la que no usaremos.

df = df.drop(["developer","release_date","game"], axis=1)

#separamos en train y test
test = df.tail(1000)
df = df.drop(df.tail(1000).index)
train = df.copy()

guardar_datos(df,"reviews", "procesed")
guardar_datos(train, "train","train")
guardar_datos(test, "test","test")

regression(train, test, "user_score")
polynominal(train, test, "user_score")

#solo sacamos el modelo final
grid_rfr(train, test, "user_score")