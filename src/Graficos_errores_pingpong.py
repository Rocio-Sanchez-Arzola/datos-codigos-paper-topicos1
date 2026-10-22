#!/usr/bin/env python
# coding: utf-8

# # Pelota de Voleybol

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')


# ## Modelado y simulación de la trayectoria del balón

# ### Parámetros
# 

# In[14]:


ARCHIVO = "datos/pingpongnuevo(benja).csv"

m   = 0.00279      # masa [kg]
D   = 0.042      # diámetro [m]
A   = np.pi*(D/2)**2     # área transversal [m^2]
C   = 0.48         # coeficiente de arrastre (pingpong)
rho = 1.225       # densidad del aire [kg/m^3]
g   = 9.8        # aceleración de gravedad [m/s^2]
B   = 0.5*C*rho*A   # Ec. (3)

FRAC_AJUSTE = 0.4  
DT = 0.0005        

REFERENCIA_PIXELES = 767
REFERENCIA_METROS = 1.75


# ### Procesamiento del CSV

# In[15]:


df = pd.read_csv(ARCHIVO)
df.columns = ["t", "x_px", "y_px", "vx_app", "vy_app"]  

t = df["t"].to_numpy() - df["t"].iloc[0]
x_px = df["x_px"].to_numpy() - df["x_px"].iloc[0]        
y_px = df["y_px"].to_numpy() - df["y_px"].iloc[0]

PX_POR_M = REFERENCIA_PIXELES / REFERENCIA_METROS

print(f"Escala: {PX_POR_M:.1f} px/m")

x_exp = x_px / PX_POR_M
y_exp = y_px / PX_POR_M


# ### Estimación de las velocidades iniciales

# In[16]:


N = max(8, int(FRAC_AJUSTE * len(t)))
vx0 = np.polyfit(t[:N], x_exp[:N], 2)[1]
vy0 = np.polyfit(t[:N], y_exp[:N], 2)[1]

print(f"Puntos utilizados para el ajuste: {N}")
print(f"Velocidad inicial: vx0 = {vx0:.3f} m/s")
print(f"vy0 = {vy0:.3f} m/s")
print(f"|v0| = {np.hypot(vx0, vy0):.3f} m/s")


# ### Simulación: Método de Euler

# In[17]:


def simular(vx0, vy0, t_max, con_roce=True, dt=DT, b_val=None):
    n = int(np.ceil(t_max / dt)) + 1
    ts = np.arange(n) * dt
    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)

    vx[0], vy[0] = vx0, vy0
    b = (B if b_val is None else b_val) if con_roce else 0.0

    for i in range(n - 1):
        v = np.hypot(vx[i], vy[i])
        ax = -b * v * vx[i] / m
        ay = -b * v * vy[i] / m - g
        x[i+1] = x[i] + vx[i] * dt
        y[i+1] = y[i] + vy[i] * dt
        vx[i+1] = vx[i] + ax * dt
        vy[i+1] = vy[i] + ay * dt
    return ts, x, y

t_max = t[-1]
ts_r, x_r, y_r = simular(vx0, vy0, t_max, con_roce=True)
ts_s, x_s, y_s = simular(vx0, vy0, t_max, con_roce=False)


# ### Gráfico

# In[18]:


fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(x_exp, y_exp, "o", ms=4, color="tab:blue", label="Experimental (Vernier)")
ax.plot(x_r, y_r, "-", lw=2, color="tab:red", label=f"Predicción con roce (C={C})")
ax.plot(x_s, y_s, "--", lw=2, color="tab:green", label="Predicción sin roce")
ax.set_xlabel("Distancia X (m)", fontsize=14)
ax.set_ylabel("Distancia Y (m)", fontsize=14)
ax.set_title("Trayectoria de una pelota de ping-pong: experimento vs. simulación")
ax.set_aspect("equal", adjustable="datalim")
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig("trayectoria_pingpong.png", dpi=300)
plt.show()


# ### Prueba de convergencia temporal de Euler

# In[19]:


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


# ### Propagación de errores y gráficos

# In[20]:


# PROPAGACIÓN DE ERRORES: PING-PONG
# Reutiliza m, D, A, C, rho, B, t, x_exp, y_exp, vx0, vy0 y simular().
# Ejecutar inmediatamente después de la celda de trayectoria de Ping-pong.

# 1. Incertidumbres medidas o asignadas
delta_m = 5e-05  # incertidumbre de masa [kg], no incluida en la banda de B
delta_d = 0.001  # incertidumbre del diámetro [m]
delta_C = 0.02  # incertidumbre del coeficiente de arrastre
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

print("INCERTIDUMBRES Y ERRORES: PING-PONG")
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
ax1.set_title("Trayectoria: Ping-pong")
ax1.set_aspect("equal", adjustable="datalim")
ax1.legend(fontsize=8)
xm = x_exp[np.argmax(y_exp)]
ym = np.max(y_exp)
ax2.set_xlim(xm - 0.2, xm + 0.2)
ax2.set_ylim(ym - 0.15, ym + 0.10)
ax2.set_title("Ampliación")
fig.tight_layout()
fig.savefig("propagacion_pingpong.png", dpi=300)
plt.show()

