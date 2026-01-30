import sys
import os
import socket
import struct
import numpy as np
import cv2
import time

# --- CONFIGURACIÓN ---
CLUSTER_IP = '127.0.0.1' # Ajusta a la IP de tu Cluster
CLUSTER_PORT = 50005
DELTA_T = 10000 

# Carga Metavision
HOME = os.path.expanduser("~")
OPENEB_PATH = os.path.join(HOME, "openeb")

# 1. AÑADIR LA CARPETA DE COMPILACIÓN (Aquí está metavision_hal.so)
# Sin esto, falla porque no encuentra el módulo de bajo nivel
sys.path.append(os.path.join(OPENEB_PATH, "build/py3"))

# 2. AÑADIR LAS CARPETAS DEL CÓDIGO FUENTE (Aquí está metavision_core)
# Sin esto, falla porque no encuentra los paquetes de Python
sys.path.append(os.path.join(OPENEB_PATH, "sdk/modules/core/python/pypkg"))
sys.path.append(os.path.join(OPENEB_PATH, "sdk/modules/core_ml/python/pypkg"))
sys.path.append(os.path.join(OPENEB_PATH, "sdk/modules/driver/python/pypkg"))
sys.path.append(os.path.join(OPENEB_PATH, "sdk/modules/hal/python/pypkg"))
try:
    from metavision_core.event_io import EventsIterator
except:
    sys.exit("Error: Metavision SDK no encontrado.")

def make_vis_frame(events, width, height):
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events) > 0:
        xs = np.clip(events['x'], 0, width-1)
        ys = np.clip(events['y'], 0, height-1)
        vis[ys, xs, :] = 255
    return vis

def run_sender_only():
    print(f"-> Conectando al Cluster {CLUSTER_IP}:{CLUSTER_PORT}...")
    try:
        sock_cluster = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_cluster.connect((CLUSTER_IP, CLUSTER_PORT))
        print("   [OK] Conectado. Iniciando transmisión...")
    except Exception as e:
        sys.exit(f"   [ERROR] No conecta: {e}")

    # Iniciar Cámara
    mv_iterator = EventsIterator(input_path="", delta_t=DELTA_T)
    h, w = mv_iterator.get_size()
    print(f"-> Cámara: {w}x{h}")

    try:
        for evs in mv_iterator:
            if evs.size == 0: continue

            # 1. Generar Imagen
            img = make_vis_frame(evs, w, h)

            # 2. Comprimir
            res, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not res: continue
            
            img_bytes = img_encoded.tobytes()
            
            # 3. Enviar [Longitud][Bytes]
            # No esperamos respuesta, solo enviamos (Fire & Forget)
            header = struct.pack('>I', len(img_bytes))
            
            try:
                sock_cluster.sendall(header + img_bytes)
            except BrokenPipeError:
                print("! El Cluster cerró la conexión.")
                break

            # Opcional: Ver lo que enviamos localmente
            cv2.imshow("Cliente Local", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        sock_cluster.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_sender_only()