# Machine-learning-project

## Introduccion

Trabajo de modelo de prediccion sobre videojuegos.

En este trabajo vamos a manipular bases de datos para poder realizar un modelo predictivo de machine learning que pueda satisfacer una necesidad y ayudar a un posible cliente, realizando prediciones sobre la nota de reviews de usuario que tendra un juego, usando datos de Metacritic.

# Resumen

Primero empece analizando una dataframe de videojuegos con el objetivo de desarrollar un modelo para poder predecir el precio de un videojuego segun sus caracteristicas.

Sin embargo, tras realizar un exhaustivo trabajo de Feature engineering y limpieza de datos, no se pudieron encontrar correlaciones suficientes para que el modelo pudiese aprender correctamente, por lo que se decidio cambiar la perspectiva.

Se cambio a una database de juegos y caracteristicas sacadas de la pagina de Metacritic, se realizo limpieza y se pudo encontrar correlaciones sufientes para construir un buen modelo de prediccion.

Para la presentación, se realizo un streamlit con el objetivo de poder probar los modelos y ver sus porcentajes de precisión y margen de error.

# Objetivo

Con un modelo de predicciones de reviews, se pueden dar los datos de un videojuego, estimaciones y mas caracteristicas, de esta forma se puede saber si la nota que le daran los usuarios al juego, con ello, ayudamos al cliente a poder realizar un estudio bien rapido, ahorrando tiempo y recursos que pueden destinar a otros apartados.

# Estructura.

- app: Donde encontraras la app de streamlit.
- Data: donde estan todos los datos, separados por raw, limpiados, train y test para el modelo de machine learning.
- Docs: Imagenes u otros archivos usados para presentación o diseño de app.
- Models: Aqui estan todos los modelos usados y guardados para poder ser importados.
- Notebooks: Todos los jupyter usados en el trabajo.
- Los archivos .py para poder ejecutar toda la limpieza de datos y creación de modelos.

