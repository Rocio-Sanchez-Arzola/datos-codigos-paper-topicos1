#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')


# ## Pelota de Voley (Referencia Benja)

# ### Parámetros

# In[8]:


ARCHIVO = "datos/volveynuevo2(benja).csv"

m    = 0.279      # masa del balón [kg]
D    = 0.203      # diámetro del balón [m]
A    = np.pi*(D/2)**2   # área transversal [m^2]
C    = 0.4       # coef. de arrastre (pelota de voley)
rho  = 1.225      # densidad del aire [kg/m^3]
g    = 9.8        # aceleración de gravedad [m/s^2]
B    = 0.5 *C*rho*A    # Parámetro de arrastre (Ec. (3) del artículo de reproducción)

FRAC_AJUSTE = 0.4  #Se utilizará el 40% inicial de los datos para estimar las velocidades iniciales
DT = 0.0005       # intervalo de tiempo entre cada iteración para Euler

REFERENCIA_METROS = 1.75 #Longitud real de la persona que lanza el balón, usado como calibración
REFERENCIA_PIXELES = 560 #Longitud de la misma referencia pero en píxeles


# ### Procesamiento del CSV

# In[9]:


df = pd.read_csv(ARCHIVO)   #lectura del archivo CSV
df.columns = ["t", "x_px", "y_px", "vx_app", "vy_app"]   #las columnas dicen (m) pero son píxeles, así que se ajustan los nombres

t = df["t"].to_numpy() - df["t"].iloc[0]                 # Convierte los tiempos a un arreglo y resta el tiempo para comenzar en t=0
x_px = df["x_px"].to_numpy() - df["x_px"].iloc[0]        # Convierte las posiciones en x a un arreglo y establece x inicial = 0
y_px = df["y_px"].to_numpy() - df["y_px"].iloc[0]       # Convierte las posiciones en x a un arreglo y establece y inicial = 0

PX_POR_M = REFERENCIA_PIXELES / REFERENCIA_METROS #Calcula cuántos píxeles de la imagen equivalen a un metro
print(f"Escala: {PX_POR_M:.1f} px/m") #Imprime la escala de conversión con 1 decimal

x_exp = x_px / PX_POR_M # convierte las posiciones en x de píxeles a metros
y_exp = y_px / PX_POR_M # convierte las posiciones en y de píxeles a metros


# ### Estimación de las velocidades iniciales

# In[10]:


N = max(8, int(FRAC_AJUSTE * len(t)))  # Calcula el 40% del número total de mediciones usando mínimo 8 puntos
vx0 = np.polyfit(t[:N], x_exp[:N], 2)[1]  #Estima la velocidad horizontal inicial mediante un ajuste cuadrático
vy0 = np.polyfit(t[:N], y_exp[:N], 2)[1]  #Estima la velocidad vertical inicial mediante un ajuste cuadrático

print(f"Puntos utilizados para el ajuste: {N}")  #Muestra cuántos puntos experimentales se usaron para estimar las velocidades
print(f"Velocidad inicial horizontal: vx0 = {vx0:.3f} m/s")  #Muestra la velocidad horizontal inicial con 3 decimales
print(f"Velocidad inicial vertical: vy0 = {vy0:.3f} m/s")  #Muestra la velocidad vertical inicial con 3 decimales
print(f"|v0| = {np.hypot(vx0, vy0):.3f} m/s")  #Calcula y muestra la magnitud de la velocidad inicial


# ### Simulación: Método de Euler

# In[11]:


def simular(vx0, vy0, t_max, con_roce=True, dt=DT, b_val=None):
    n = int(t_max / dt) + 1   #calcula la cantidad de puntos necesarios para superar el tiempo final
    ts = np.arange(n) * dt   #genera los instantes de tiempo de la simulación, separados por el intervalo dt
    x = np.zeros(n); y = np.zeros(n)   #Crea un arreglo para almacenar las posiciones, comenzando con todos los valores en cero
    vx = np.zeros(n); vy = np.zeros(n)   #Crea un arreglo para almacenar las velocidades de cada iteración

    vx[0], vy[0] = vx0, vy0   #Asigna las velocidades iniciales estimadas al primer elemento de los arreglos
    b = (B if b_val is None else b_val) if con_roce else 0.0  #Utiliza el parámetro de arrastre B si hay roce, 0 si no hay roce

    for i in range(n - 1):  
        v = np.hypot(vx[i], vy[i]) #Calcula la magnitud de la velocidad en el instante actual
        ax = -b* v * vx[i] / m    #Calcula la aceleración horizontal producido por la resistencia del aire
        ay = -b * v * vy[i] / m - g   #Calcula la aceleración vertical considerando la resistencia del aire y la gravedad
        x[i+1]  = x[i]  + vx[i] * dt    #Calcula la siguiente posición horizontal
        y[i+1]  = y[i]  + vy[i] * dt    #Calcula la siguiente posición vertical 
        vx[i+1] = vx[i] + ax * dt       #Calcula la siguiente velocidad horizontal
        vy[i+1] = vy[i] + ay * dt       #Calcula la siguiente velocidad vertical
    return ts, x, y #Entrega los tiempos y las posiciones horizontales y verticales calculadas

t_max = t[-1] #Obtiene el último tiempo experimental para establecer la duración de la simulación
ts_r, x_r, y_r = simular(vx0, vy0, t_max, con_roce=True) #Ejecuta Euler con resistencia del aire
ts_s, x_s, y_s = simular(vx0, vy0, t_max, con_roce=False) #Ejecuta Euler sin resistencia del aire


# ### Gráfico

# In[12]:


fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(x_exp, y_exp, "o", ms=4, color="tab:blue", label="Experimental (Vernier)")
ax.plot(x_r, y_r, "-", lw=2, color="tab:red", label=f"Predicción con roce (C={C})")
ax.plot(x_s, y_s, "--", lw=2, color="tab:green", label="Predicción sin roce")
ax.set_xlabel("Distancia X (m)")
ax.set_ylabel("Distancia Y (m)")
ax.set_title("Trayectoria de un balón de vóleibol: experimento vs. simulación")
ax.set_aspect("equal", adjustable="datalim")
fontsize=16
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=13)
ax.legend()
fig.tight_layout()
fig.savefig("trayectoria_voley.png", dpi=300)
plt.show()


# ### Convergencia temporal Euler

# In[13]:


# ----------------------------------------------------------------------------
# PRUEBA DE CONVERGENCIA TEMPORAL DE EULER
# ----------------------------------------------------------------------------

pasos = [0.004, 0.002, 0.001, 0.0005, 0.00025]

resultados = []

# Simular con distintos pasos temporales
for dt in pasos:

    ts, xs, ys = simular(
        vx0, vy0, t_max,
        con_roce=True,
        dt=dt
    )

    resultados.append((ts, xs, ys))

# Comparar trayectorias sucesivas
diferencias = []

for i in range(len(resultados) - 1):

    ts1, x1, y1 = resultados[i]
    ts2, x2, y2 = resultados[i + 1]

    # Interpolar la solución más fina sobre los tiempos de la gruesa
    x2_interp = np.interp(ts1, ts2, x2)
    y2_interp = np.interp(ts1, ts2, y2)

    # Diferencia RMS entre ambas trayectorias
    D = np.sqrt(np.mean(
        (x1 - x2_interp)**2 +
        (y1 - y2_interp)**2
    ))

    diferencias.append(D)

    print(
        f"dt = {pasos[i]:.5f} vs {pasos[i+1]:.5f} s "
        f"--> Diferencia RMS = {D:.6f} m"
    )

# Orden de convergencia observado
print("\nORDEN DE CONVERGENCIA")

for i in range(len(diferencias) - 1):

    if diferencias[i+1] > 0:

        p = np.log2(
            diferencias[i] / diferencias[i+1]
        )

        print(
            f"Entre refinamientos {i+1} y {i+2}: "
            f"p = {p:.3f}"
        )


# ### Propagación de incertidumbres

# In[14]:


# 1. Incertidumbres medidas o asignadas
delta_m = 0.001  # incertidumbre de masa [kg], no incluida en la banda de B
delta_d = 0.001  # incertidumbre del diámetro [m]
delta_C = 0.05  # incertidumbre del coeficiente de arrastre
delta_p = 2.0        # incertidumbre de un clic del seguimiento [px]
delta_A = A * 2 * delta_d / D

# Estimación relativa de delta_rho según precisión de P y T.
# rho nominal permanece igual al notebook principal (1.225 kg/m³).
P, delta_P = 100080.0, 10.0  # presión y su incertidumbre [Pa]
Temp_C, delta_T = 15.0, 1.0  # temperatura [°C] e incertidumbre [K]
delta_rho = rho * np.sqrt((delta_P / P)**2
                          + (delta_T / (Temp_C + 273.15))**2)

# 2. Incertidumbre de B = C*rho*A/2 (variables asumidas independientes)
delta_B = B * np.sqrt((delta_C / C)**2 + (delta_rho / rho)**2
                      + (delta_A / A)**2)
B_min, B_max = B - delta_B, B + delta_B

# 3. Incertidumbre aproximada de la escala y las posiciones relativas
err_rel_S = np.sqrt(2) * delta_p / REFERENCIA_PIXELES
error_px_m = np.sqrt(2) * delta_p / PX_POR_M
dx_exp = np.sqrt(error_px_m**2 + (x_exp * err_rel_S)**2)
dy_exp = np.sqrt(error_px_m**2 + (y_exp * err_rel_S)**2)

# 4. Incertidumbre estadística de las velocidades (mismo np.polyfit)
# Las velocidades nominales vx0, vy0 ya están calculadas en la celda anterior.
p_x, cov_x = np.polyfit(t[:N], x_exp[:N], 2, cov=True)
p_y, cov_y = np.polyfit(t[:N], y_exp[:N], 2, cov=True)
delta_vx0 = np.sqrt(cov_x[1, 1])
delta_vy0 = np.sqrt(cov_y[1, 1])
v0 = np.hypot(vx0, vy0)
delta_v0 = (np.hypot(vx0 * delta_vx0, vy0 * delta_vy0) / v0
            if v0 > 0 else np.nan)

# 5. Simulaciones para los extremos de B (sin recalcular la nominal)
_, x_bajo, y_bajo = simular(vx0, vy0, t_max, b_val=B_min)
_, x_alto, y_alto = simular(vx0, vy0, t_max, b_val=B_max)

# Comparación a la misma coordenada X: requiere trayectorias crecientes en X.
if np.all(np.diff(x_bajo) > 0) and np.all(np.diff(x_alto) > 0):
    y_1 = np.interp(x_r, x_bajo, y_bajo, left=np.nan, right=np.nan)
    y_2 = np.interp(x_r, x_alto, y_alto, left=np.nan, right=np.nan)
else:
    # Si X no es monótona, se usan las alturas a tiempos equivalentes.
    y_1, y_2 = y_bajo, y_alto
y_lim_inf, y_lim_sup = np.minimum(y_1, y_2), np.maximum(y_1, y_2)

# 6. RMSE: comparación de posiciones simuladas y experimentales al mismo t
def rmse_2d(ts, xs, ys):
    xi = np.interp(t, ts, xs)
    yi = np.interp(t, ts, ys)
    return np.sqrt(np.mean((xi - x_exp)**2 + (yi - y_exp)**2))

def error_porcentual(real, simulado):
    return 100 * abs(real - simulado) / abs(real) if real != 0 else np.nan

print("INCERTIDUMBRES Y ERRORES: VÓLEIBOL")
print(f"B = {B:.6f} ± {delta_B:.6f} kg/m")
print(f"Incertidumbre relativa de escala: {100*err_rel_S:.2f} %")
print(f"Incertidumbre promedio X: {1000*np.mean(dx_exp):.2f} mm")
print(f"Incertidumbre promedio Y: {1000*np.mean(dy_exp):.2f} mm")
print(f"vx0 = {vx0:.3f} ± {delta_vx0:.3f} m/s")
print(f"vy0 = {vy0:.3f} ± {delta_vy0:.3f} m/s")
print(f"|v0| = {v0:.3f} ± {delta_v0:.3f} m/s")
print(f"RMSE con roce: {rmse_2d(ts_r, x_r, y_r):.4f} m")
print(f"RMSE sin roce: {rmse_2d(ts_s, x_s, y_s):.4f} m")
print(f"Error de alcance: {error_porcentual(np.max(x_exp), np.max(x_r)):.2f} %")
print(f"Error de altura máxima: {error_porcentual(np.max(y_exp), np.max(y_r)):.2f} %")

# 7. Gráfico completo y ampliación de la banda B ± delta_B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for ax in (ax1, ax2):
    ax.fill_between(x_r, y_lim_inf, y_lim_sup, alpha=0.20,
                    color="tab:red", label=r"Banda de $B\pm\delta B$")
    ax.errorbar(x_exp, y_exp, xerr=dx_exp, yerr=dy_exp, fmt="o",
                ms=2.5, color="tab:blue", ecolor="black", capsize=1.2,
                errorevery=2, label="Experimental (Vernier)")
    ax.plot(x_r, y_r, color="tab:red", label="Con roce")
    ax.plot(x_s, y_s, "--", color="tab:green", label="Sin roce")
    ax.set_xlabel("Distancia X (m)")
    ax.set_ylabel("Distancia Y (m)")
    ax.grid(alpha=0.3)
ax1.set_title("Trayectoria: Vóleibol")
ax1.set_aspect("equal", adjustable="datalim")
ax1.legend(fontsize=8)
xm = x_exp[np.argmax(y_exp)]
ym = np.max(y_exp)
ax2.set_xlim(xm - 0.35, xm + 0.35)
ax2.set_ylim(ym - 0.25, ym + 0.10)
ax2.set_title("Ampliación")
fig.tight_layout()
fig.savefig("propagacion_voley.png", dpi=300)
plt.show()

