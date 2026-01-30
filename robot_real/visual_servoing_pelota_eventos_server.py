import sys
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import socket  # [Nuevo] Para TCP
import warnings

# --- Limpieza de terminal ---
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
 
# ================= 0. CONFIGURACIÓN DE RUTAS =================
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
    print(f"Librería Metavision cargada.")
except ImportError as e:
    sys.exit(f"Error importando Metavision. \nIntenta ejecutar en la terminal: source {OPENEB_ROOT}/build/utils/scripts/setup_env.sh\nDetalle: {e}")
 
# ================= 1. CONFIGURACIÓN =================
REPO_PATH = os.path.join(os.getcwd(), 'SLTNet-v1.0-main')
 
# AJUSTES DE CÁMARA E IMAGEN
DELTA_T = 100000        
TARGET_HW = (1280, 720 ) 
TARGET_HW = (640, 480 )
 
#TARGET_HW = (256,256)
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# IMPORTANTE: Definición de colores para visualización (BGR para OpenCV)
# Index 0: Negro (Fondo)
# Index 1: Azul
# Index 2: Rojo
# Index 3: Verde (Pelota)
COLORS = np.array([
    [0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]
], dtype=np.uint8)
 
# Identificador de la clase que queremos encuadrar (el verde es el índice 3)
BALL_CLASS_ID = 3
 
# ================= 2. MODELO =================
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)
 
try:
    from SLTNet import SLTNet
    print("Modelo SLTNet cargado correctamente desde la raíz.")
except ImportError:
    try:
        from model.SLTNet import SLTNet
        print("Modelo SLTNet cargado desde subcarpeta model.")
    except ImportError as e:
        print(f"Ruta buscada: {REPO_PATH}")
        sys.exit(f"Error crítico: No se puede importar SLTNet. Detalles: {e}")
 
from spikingjelly.activation_based import functional
 
def load_model_weights(model, weight_path):
    print(f"Cargando pesos: {os.path.basename(weight_path)}")
    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        print(f"Aviso carga: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    return model
 
# ================= 3. PROCESAMIENTO =================
def events_to_tensor(events, width, height):
    on_map = np.zeros((height, width), dtype=np.float32)
    off_map = np.zeros((height, width), dtype=np.float32)
 
    mask = (events['x'] < width) & (events['y'] < height)
    evs = events[mask]
 
    on_evs = evs[evs['p'] == 1]
    off_evs = evs[evs['p'] == 0]
 
    if len(on_evs) > 0: on_map[on_evs['y'], on_evs['x']] = 1.0
    if len(off_evs) > 0: off_map[off_evs['y'], off_evs['x']] = 1.0
 
    return on_map, off_map
 
def make_vis_frame(events, width, height):
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events) > 0:
        xs = np.clip(events['x'], 0, width-1)
        ys = np.clip(events['y'], 0, height-1)
        vis[ys, xs, :] = 255
    return vis

# ================= CONFIGURACIÓN TCP =================
TCP_IP = '127.0.0.1'  # Localhost
TCP_PORT = 5005
BUFFER_SIZE = 1024

def start_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    print(f"Espera conexión en {TCP_IP}:{TCP_PORT}...")
    conn, addr = s.accept()
    print(f"Conectado con: {addr}")
    return s, conn

# ================= 4. MAIN MODIFICADO =================
def run_camera_inference_server(weight_path):
    # 1. Iniciar Servidor TCP
    server_socket, conn = start_server()

    # 2. Cargar Modelo (Igual que antes)
    try:
        net = SLTNet().to(DEVICE)
        net = load_model_weights(net, weight_path)
        net.eval()
        functional.set_step_mode(net, 'm')
    except Exception as e:
        sys.exit(f"Error modelo: {e}")

    # 3. Iniciar Cámara Eventos
    try:
        mv_iterator = EventsIterator(input_path="", delta_t=DELTA_T)
    except Exception as e:
        print(f"Error cámara: {e}")
        return

    height, width = mv_iterator.get_size()
    print(f"-> Cámara: {width}x{height}")

    # Warmup...
    if DEVICE.type == 'cuda':
        dummy = torch.zeros(1, 2, TARGET_HW[0], TARGET_HW[1]).to(DEVICE)
        with torch.no_grad(): net(dummy)
        functional.reset_net(net)

    print("-> INFERENCIA + TCP ACTIVO.")
    try:
        for evs in mv_iterator:
            if evs.size == 0: continue

            # ... (Preprocesamiento e Inferencia SNN igual que antes) ...
            on_ch, off_ch = events_to_tensor(evs, width, height)
            input_tensor = torch.from_numpy(np.stack([on_ch, off_ch], axis=0)).unsqueeze(0).to(DEVICE)
            input_resized = F.interpolate(input_tensor, size=TARGET_HW, mode='nearest')

            with torch.no_grad():
                functional.reset_net(net)
                output = net(input_resized)
                if isinstance(output, (tuple, list)): output = output[0]
                if output.dim() == 5: output = output.mean(0)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # ... (Visualización igual que antes) ...
            bg = make_vis_frame(evs, width, height)
            pred_big = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # --- DETECCIÓN Y ENVÍO TCP ---
            ball_mask = np.uint8(pred_big == BALL_CLASS_ID)
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Encontrar el contorno más grande (la pelota principal)
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_cnt) > 50:
                    x, y, w, h = cv2.boundingRect(largest_cnt)
                    
                    # Calcular las 4 esquinas (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
                    tl = (x, y)
                    tr = (x + w, y)
                    br = (x + w, y + h)
                    bl = (x, y + h)

                    # Formatear mensaje: "tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y"
                    message = f"{tl[0]},{tl[1]},{tr[0]},{tr[1]},{br[0]},{br[1]},{bl[0]},{bl[1]}\n"
                    
                    try:
                        # print(f"DEBUG Server: Enviando BBox sobre resolucion {width}x{height} -> {message.strip()}")
                        conn.send(message.encode())
                    except BrokenPipeError:
                        print("Cliente desconectado.")
                        break

                    # Dibujar para debug local
                    cv2.rectangle(bg, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Visualización Local (Opcional, puede comentarse para velocidad)
            cv2.imshow("Server View", bg)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_WEIGHTS = "best.pth"
    if os.path.exists(MODEL_WEIGHTS):
        run_camera_inference_server(MODEL_WEIGHTS)