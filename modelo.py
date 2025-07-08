# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import pandas as pd

# ------------------------
# PARÁMETROS GENERALES
# ------------------------
N = 1000        # población total
I0 = 10         # infectados iniciales
R0 = 0          # recuperados iniciales
S0 = N - I0 - R0
beta = 0.3      # tasa de transmisión base
gamma = 0.1     # tasa de recuperación
dias = 160      # duración de la simulación

# ------------------------
# 1. MODELO SIR (ODE)
# ------------------------

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Tiempo y condiciones iniciales
t = np.linspace(0, dias, dias)
y0 = S0, I0, R0
sol = odeint(sir_model, y0, t, args=(N, beta, gamma))
S_sir, I_sir, R_sir = sol.T

# ------------------------
# 2. MODELO ABM (simulación)
# ------------------------

# Crear agentes con estado y cumplimiento de medidas
agentes = [
    {
        'estado': 'I' if i < I0 else 'S',
        'cumple_medidas': random.uniform(0.3, 1.0)
    }
    for i in range(N)
]

historial_abm = []

for dia in range(dias):
    nuevos_infectados = []

    for i, agente in enumerate(agentes):
        if agente['estado'] == 'S':
            contactos = random.sample(agentes, 10)
            for contacto in contactos:
                if contacto['estado'] == 'I':
                    riesgo = beta * (1 - agente['cumple_medidas'])
                    if random.random() < riesgo:
                        nuevos_infectados.append(i)
                        break

        elif agente['estado'] == 'I':
            if random.random() < gamma:
                agente['estado'] = 'R'

    for idx in nuevos_infectados:
        agentes[idx]['estado'] = 'I'

    # Registrar el estado de cada día
    estados = [ag['estado'] for ag in agentes]
    historial_abm.append({
        'Día': dia,
        'Susceptibles': estados.count('S'),
        'Infectados': estados.count('I'),
        'Recuperados': estados.count('R')
    })

# Convertir a DataFrame
df_abm = pd.DataFrame(historial_abm)

# ------------------------
# 3. GRAFICAR RESULTADOS
# ------------------------

plt.figure(figsize=(12, 6))

# SIR
plt.plot(t, S_sir, 'b-', label='SIR - Susceptibles')
plt.plot(t, I_sir, 'r-', label='SIR - Infectados')
plt.plot(t, R_sir, 'g-', label='SIR - Recuperados')

# ABM
plt.plot(df_abm['Día'], df_abm['Susceptibles'], 'b--', label='ABM - Susceptibles')
plt.plot(df_abm['Día'], df_abm['Infectados'], 'r--', label='ABM - Infectados')
plt.plot(df_abm['Día'], df_abm['Recuperados'], 'g--', label='ABM - Recuperados')

plt.xlabel("Días")
plt.ylabel("Cantidad de personas")
plt.title("Comparación entre Modelo SIR y ABM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
