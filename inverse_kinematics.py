import sys
import tempfile
import json

import numpy as np
import matplotlib.pyplot as plt

# listas para coletar dados
time_log = []
s_log = []
sdot_log = []
sddot_log = []

try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

import math
from controller import Supervisor

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')


IKPY_MAX_ITERATIONS = 4

# Initialize the Webots Supervisor.
supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# Create the arm chain from the URDF
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))
armChain = Chain.from_urdf_file(filename, active_links_mask=[False, True, True, True, True, True, True, False])

# Initialize the arm motors and encoders.
motors = []
for link in armChain.links:
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)

# Get the arm and target nodes.
target = supervisor.getFromDef('TARGET')
arm = supervisor.getSelf()


print('Move the yellow and black sphere to move the arm...')

# Duração padrão da interpolação (segundos)
T = 1.0
t = 0.0

# alvo anterior (para detectar mudanças)
prev_target = [0, 0, 0]

# junta inicial q0
q0 = [m.getPositionSensor().getValue() for m in motors]
qf = q0[:]  # final inicial igual ao atual

# Limitando o tempo da simulação
total_time = 0.0
limit = 4.0


# ---------------------------
# 1. Escalonamento Cúbico
# ---------------------------
def s_cubico(t,T):

    s = 3*(t/T)**2 - 2*(t/T)**3
    
    return s


def s_cubico_derivadas(t,T):
    
    s_dot = (6/T**2)*t - (6/T**3)*(t**2)
    s_ddot = (6/T**2) - (12/T**3)*t
    
    return s_dot, s_ddot

# ---------------------------
# 2. Escalonamento Quintico
# ---------------------------    
def s_quintico(t, T):

    tau = t / T
    if tau > 1: tau = 1
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    
    return s

def s_quintico_derivadas(t,T):
    
    tau = t / T
    
    s_dot = (30/T) * (tau**2) - (60/T) * (tau**3) + (30/T) * (tau**4)

    s_ddot = (60/T**2) * tau - (180/T**2) * (tau**2) + (120/T**2) * (tau**3)

    return s_dot, s_ddot

# ---------------------------------
# 3. Escalonamento Trapezoidal
# ---------------------------------
def s_trapezoidal(t, T, tb_frac_local=0.25):
   
    tb = tb_frac_local * T
    if T <= 0:
        return 1.0
    a = 1.0 / (tb * (T - tb))
    v = a * tb

    # limites
    if t <= 0:
        return 0.0
    if t >= T:
        return 1.0

    # fases
    if t <= tb:
        return 0.5 * a * t**2
    elif t <= (T - tb):
        s_tb = 0.5 * a * tb**2
        s = s_tb + v * (t - tb)
        return s
    else:
        s_tb = 0.5 * a * tb**2
        s_lin = s_tb + v * (T - 2*tb)
        td = t - (T - tb)
        s = s_lin + v*td - 0.5*a*td**2
        return s


def s_trapezoidal_derivadas(t, T, tb_frac_local=0.25):

    if T <= 0:
        return 0.0, 0.0

    tb = tb_frac_local * T
    if tb <= 0:
        tb = 1e-8
    if tb >= T/2:
        tb = 0.49 * T

    a = 1.0 / (tb * (T - tb))
    v = a * tb

    if t <= 0 or t >= T:
        s_dot, s_ddot = 0, 0
        
        return s_dot, s_ddot

    if t <= tb:
        return a * t, a
    elif t <= (T - tb):
        s_dot, s_ddot = v, 0
        return s_dot, s_ddot
    else:
        td = t - (T - tb)
        s_dot, s_ddot = v - a * td, -a
        return s_dot, s_ddot

# ---------------------------------
# 4. Selecionador de Escalonamento
# ---------------------------------
def s_seletor(escalonamento):

    if escalonamento == "cubico":
        s_ = s_cubico
        s_derivada = s_cubico_derivadas
        
    elif escalonamento == "quintico":
        s_ = s_quintico
        s_derivada = s_quintico_derivadas
    
    elif escalonamento == "trapezoidal":
        s_ = s_trapezoidal
        s_derivada = s_trapezoidal_derivadas
    
    return s_, s_derivada


escalonamento = "trapezoidal"

s_escalonamento, s_escalonamento_derivada = s_seletor(escalonamento)


while supervisor.step(timeStep) != -1:
    dt = timeStep / 1000.0  # Webots usa ms → s
    total_time += dt
    
    if total_time >= limit:
        print(f"Fim dos {limit} segundos de simulação.")
        break
    
    # ---------------------------
    # 1. Ler posição do target
    # ---------------------------
    targetPosition = target.getPosition()
    armPosition = arm.getPosition()

    x = -(targetPosition[1] - armPosition[1])
    y =  (targetPosition[0] - armPosition[0])
    z =  (targetPosition[2] - armPosition[2])

    # Detectar mudança do target
    dist_target = math.sqrt((targetPosition[0]-prev_target[0])**2 +
                            (targetPosition[1]-prev_target[1])**2 +
                            (targetPosition[2]-prev_target[2])**2)

    # Se o alvo mudou mais que 1 cm, recalcular e reiniciar escalonamento
    if dist_target > 0.01:
        prev_target = targetPosition[:]

        #initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
        initial_position = [m.getPositionSensor().getValue() for m in motors]
        
        ikResults = armChain.inverse_kinematics([x, y, z])

        q0 = initial_position[:] 
        qf = ikResults[1:1+len(motors)][:]
        t = 0


    # ---------------------------
    # 2. Escalonamento
    # ---------------------------
    #t += dt
    #if t > T:
    #    t = T
    if t < T:
        t += dt
        if t > T:
            t = T

    # Calcular s(t)  
    s = s_escalonamento(t,T) 

    # Derivadas do escalonamento
    s_dot, s_ddot = s_escalonamento_derivada(t,T)
    

    # Posicao suavizada: q(t) = q0 + s * (qf - q0)
    q_interp = [q0_i + s*(qf_i - q0_i) for q0_i, qf_i in zip(q0, qf)]
    
    # ---------------------------
    # 3. Aplicar nos motores
    # ---------------------------
    for motor, q in zip(motors, q_interp):
        print( f"Definindo {motor.getName()} para {q:.3f} ")
        motor.setPosition(q)   
    
    # Log dos dados para gráficos
    time_log.append(t)
    s_log.append(s)
    sdot_log.append(s_dot)
    sddot_log.append(s_ddot)
    
        
print("Gerando gráficos reais do escalonamento...")

# --- Gráfico s(t)
plt.figure()
plt.plot(time_log, s_log)
plt.xlabel("t (s)")
plt.ylabel("s(t)")
plt.title(f"Escalonamento {escalonamento} Real s(t)")
plt.grid(True)
plt.savefig("s_t_real.png")
plt.close()

# --- Gráfico ṡ(t)
plt.figure()
plt.plot(time_log, sdot_log)
plt.xlabel("t (s)")
plt.ylabel("ṡ(t)")
plt.title(f"Velocidade Real do Escalonamento {escalonamento}")
plt.grid(True)
plt.savefig("s_dot_t_real.png")
plt.close()

# --- Gráfico s̈(t)
plt.figure()
plt.plot(time_log, sddot_log)
plt.xlabel("t (s)")
plt.ylabel("s̈(t)")
plt.title(f"Aceleração Real do Escalonamento {escalonamento}")
plt.grid(True)
plt.savefig("s_ddot_t_real.png")
plt.close()

print("Gráficos gerados: s_t_real.png, s_dot_t_real.png, s_ddot_t_real.png")

with open("time_log.txt", "w", encoding="utf-8") as f:
    f.write(str(time_log))
    f.close()

with open(f"{escalonamento}_s_log.txt", "w", encoding="utf-8") as f:
    f.write(str(s_log))
    f.close()

with open(f"{escalonamento}_sdot_log.txt", "w", encoding="utf-8") as f:
    f.write(str(sdot_log))
    f.close()

with open(f"{escalonamento}_sddot_log.txt", "w", encoding="utf-8") as f:
    f.write(str(sddot_log))
    f.close()
print("Arquivos de dados gerados: s_log.txt, sdot_log.txt, sddot_log.txt")