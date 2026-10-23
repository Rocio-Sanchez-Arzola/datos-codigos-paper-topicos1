# datos-codigos-paper-topicos1

Este repositorio contiene los códigos para las simulaciones, cálculo de la propagación de incertidumbres y los datos experimentales usados para las simulaciones mencionadas en el artículo *"Reproducción experimental de 'Sports, Smartphones, and Simulation as an Engaging Method to Teach Projectile Motion Incorporating Air Resistance'"*.

---

## Archivos del Repositorio

### Códigos 

- **`Graficos_errores_pingpong.ipynb`**: procesamiento de datos, estimación de velocidad inicial, simulaciones y propagación de incertidumbres de la pelota de ping-pong.
- **`Graficos_errores_tenis.ipynb`**: mismo análisis para la pelota de tenis.
- **`Graficos_errores_voley.ipynb`**: mismo análisis para la pelota de vóleibol.

### Datos experimentales

- **`datos/pingpong.csv`**: registro experimental de la pelota de ping-pong.
- **`datos/tenis.csv`**: registro experimental de la pelota de tenis.
- **`datos/voley.csv`**: registro experimental de la pelota de vóleibol.

### Tablas de Resultados 

- **`velocidades_iniciales.csv`**: componentes de la velocidad inicial \(v_{x0}\), \(v_{y0}\), magnitud \(|v_0|\) e incertidumbre.
- **`comparacion_modelos.csv`**: parámetro \(B\), incertidumbre \(\delta B\) y RMSE de los modelos con y sin roce.
- **`convergencia_temporal.csv`**: comparación de pasos temporales sucesivos, diferencias RMS entre simulaciones y orden de convergencia.

Estas tablas resumen los valores obtenidos mediante los códigos, no contienen resultados adicionales.

##  Requisitos e Instalación

El código está escrito en **Python 3** haciendo uso de las siguientes librerías:

- `numpy`
- `pandas`
- `matplotlib`
- Jupyter Notebook o Google Colab para abrir los archivos `.ipynb`

Instalación de las librerías principales:

```bash
pip install numpy pandas matplotlib notebook
```

---

##  Instrucciones de Ejecución

1. Descargar o clonar el repositorio.
2. Comprobar que los archivos experimentales estén en la carpeta **`datos/`**, tal como indican las rutas `ARCHIVO` de los notebooks.
3. Abrir el notebook correspondiente a la pelota que se desea analizar.
4. Ejecutar las celdas en orden, desde el inicio.
5. Revisar los resultados numéricos, los gráficos de trayectoria e incertidumbre y el análisis de convergencia.

