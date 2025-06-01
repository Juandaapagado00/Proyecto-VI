import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import serial
import threading
import time

# Modo Simulacion - Real
modo_simulacion = False
puerto_serial = 'COM12'
baudrate = 115200

max_adc = 4095
voltaje_referencia = 3.3

sample_rate = 250
duration = 2
fft_window_seconds = 0.5

buffer_size = int(sample_rate * duration)
fft_window_size = int(sample_rate * fft_window_seconds)

data_buffer = []
x_vals = []
t = 0

contador_muestras = 0
muestreo_real = 0

lock = threading.Lock()

if not modo_simulacion:
    ser = serial.Serial(puerto_serial, baudrate, timeout=1)

# Clasificacion de onda
def clasificar_onda(frecuencia):
    if 0.5 <= frecuencia < 4:
        return "Delta (sueño profundo)"
    elif 4 <= frecuencia < 8:
        return "Theta (meditación/somnolencia)"
    elif 8 <= frecuencia < 13:
        return "Alpha (relajación)"
    elif 13 <= frecuencia < 30:
        return "Beta (actividad mental)"
    elif 30 <= frecuencia <= 100:
        return "Gamma (alta concentración)"
    else:
        return "Fuera de rango EEG"

# Adquisicion de datos
def adquirir_datos():
    global t, contador_muestras
    while True:
        if modo_simulacion:
            val = int(2048 + 1000 * np.sin(2 * np.pi * 10 * t) + 100 * np.random.randn())
        else:
            line = ser.readline().decode(errors='ignore').strip()
            if not line.isdigit():
                continue
            val = int(line)

        t += 1 / sample_rate
        volt = (val / max_adc) * voltaje_referencia

        with lock:
            x_vals.append(t)
            data_buffer.append(volt)
            contador_muestras += 1

            if len(data_buffer) > buffer_size:
                data_buffer[:] = data_buffer[-buffer_size:]
                x_vals[:] = x_vals[-buffer_size:]

# Muestreo real
def mostrar_muestreo():
    global contador_muestras
    while True:
        time.sleep(1)
        with lock:
            muestras = contador_muestras
            contador_muestras = 0
        print(f"[INFO] Tasa de muestreo real: {muestras} muestras/segundo")

# Iniciar hilos
threading.Thread(target=adquirir_datos, daemon=True).start()
threading.Thread(target=mostrar_muestreo, daemon=True).start()

# Visualizacion de grafica con Matplotlib
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig.canvas.manager.set_window_title("Gráfica de Señal EEG")

fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')

for ax in [ax1, ax2]:
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

line1, = ax1.plot([], [], color='cyan')
line2, = ax2.plot([], [], color='magenta')

ax1.set_title("Osciloscopio - Señal EEG")
ax1.set_ylim(0, voltaje_referencia)
ax1.set_ylabel("Voltaje (V)")

ax2.set_title("Frecuencia dominante (FFT)")
ax2.set_xlim(0, 60)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Frecuencia (Hz)")
ax2.set_ylabel("Magnitud")

# Actualizacion de la FFT
actualizar_fft_cada = 4 
contador_actualizacion_fft = 0

try:
    while True:
        with lock:
            if len(x_vals) > 1:
                line1.set_data(x_vals, data_buffer)
                ax1.set_xlim(x_vals[0], x_vals[-1])

            contador_actualizacion_fft += 1
            if len(data_buffer) >= fft_window_size and contador_actualizacion_fft >= actualizar_fft_cada:
                contador_actualizacion_fft = 0  

                ventana = np.array(data_buffer[-fft_window_size:], dtype=float)
                ventana -= np.mean(ventana)

                fft_result = fft(ventana)
                freqs = fftfreq(fft_window_size, 1 / sample_rate)
                fft_magnitude = np.abs(fft_result[:fft_window_size // 2])
                freqs = freqs[:fft_window_size // 2]

                # Filtrar las frecuencias hasta 60 Hz
                mascara = freqs <= 60
                freqs_filtradas = freqs[mascara]
                fft_magnitude_filtrada = fft_magnitude[mascara]

                line2.set_data(freqs_filtradas, fft_magnitude_filtrada)
                ax2.set_ylim(0, np.max(fft_magnitude_filtrada) + 1)

                dominant_freq = freqs_filtradas[np.argmax(fft_magnitude_filtrada)]
                tipo_onda = clasificar_onda(dominant_freq)
                ax2.set_title(f"Frecuencia dominante: {dominant_freq:.2f} Hz - {tipo_onda}", color='white')

        plt.pause(0.001)

except KeyboardInterrupt:
    print("\n[INFO] Programa terminado.")