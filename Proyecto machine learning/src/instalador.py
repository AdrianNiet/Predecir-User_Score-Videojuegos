import subprocess

# Lista de paquetes. (Se pueden añadir mas si es necesario)
packages = [
    "pandas",
    "numpy",
    "seaborn",
    "matplotlib",
    "scikit-learn",
    "ydata-profiling"
]

# Instala cada paquete.
for package in packages:
    subprocess.run(["pip", "install", package])