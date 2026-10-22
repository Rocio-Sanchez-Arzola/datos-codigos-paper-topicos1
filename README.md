# datos-codigos-paper-topicos1

Este repositorio contiene los códigos para las simulaciones, cálculo de la propagación de incertidumbres y los datos experimentales usados para las simulaciones mencionadas en el artículo *"Reproducción experimental de 'Sports, Smartphones, and Simulation as an Engaging Method to Teach Projectile Motion Incorporating Air Resistance'"*.

---

## Archivos del Repositorio

* **`propagación_de_errores_tópicos.py`**: Script de Python que calcula la densidad del aire, el parámetro de arrastre ($B$), con sus respectivas propagaciones de incertidumbres, el ajuste de los datos experimentales para determinar las velocidades iniciales de cada medición, aplicación de RMSE 2D y el error porcentual asociado al alcance horizontal y altura máxima de los lanzamientos.
* **`volveynuevo2(benja).csv`**: Datos experimentales (tiempo y posición en X e Y medida en píxeles) entregados por el software Vernier Video Analysis al trackear la trayectoria de la pelota de vóleibol.
* **`tenisnuevo2.csv`**: Datos experimentales (tiempo y posición en X e Y medida en píxeles) entregados por el software Vernier Video Analysis al trackear la trayectoria de la pelota de tenis.
* **`pingpongnuevo(benja).csv`**: Datos experimentales (tiempo y posición en X e Y medida en píxeles) entregados por el software Vernier Video Analysis al trackear la trayectoria de la pelota de pingpong.

---

##  Requisitos e Instalación

El código está escrito en **Python 3** haciendo uso de las siguientes librerías:

* `pandas`
* `numpy`
* `matplotlib`

Para instalarlas se puede ejecutar lo siguiente en la terminal:

```bash
pip install pandas numpy matplotlib scipy
```

---

##  Instrucciones de Ejecución

1. **Clonar o descargar** el repositorio asegurándote de mantener el script `.py` y los tres archivos `.csv` en la misma carpeta raíz.
2. **Ejecutar el script** desde la terminal o entorno de desarrollo:

   ```bash
   python propagación_de_errores_tópicos.py
   ```

   *(También es compatible con plataformas en la nube como Google Colab).*

3. **Resultados esperados:** La ejecución mostrará los datos presentes en el Cuadro 2 y desplegará los gráficos presentes en el artículo.
