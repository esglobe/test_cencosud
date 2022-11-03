# Inicio del proyecto

Primeramente, se recomienda generar el environment del proyecto utilizando:

1. Instalar miniconda [Página](https://docs.conda.io/en/latest/miniconda.html)
2. En terminal
 ~~~
conda env create -f ./environment.yml
conda activate ssev_analytics
conda env update --file ./environment.yml --prune
~~~

## Estructura del proyecto

~~~
 ├─ prueba_cencosud
     └─ data
     └─ ml_utils
     └─ models
     └─ prueba
~~~

**data**: Corresponde al directorio de los .csv generados en el estudio de los datos y resultados tras el entrenamiento de redes RNN.
**ml_utils**:  Módulos (librería) usadas en *ml_test_cencosud.ipynb*.
**prueba**: Archivos de la prueba Cencosud.

En el notebook *ml_test_cencosud.ipynb* se realizan los desarrollos del test.
