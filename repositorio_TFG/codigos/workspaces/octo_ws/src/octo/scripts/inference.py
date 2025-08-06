#!/usr/bin/env python3
import os
# Desactivar paralelismo en tokenizers para evitar warnings de multi-hilo
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
from octo.model.octo_model import OctoModel
import jax
import threading
from rclpy.executors import MultiThreadedExecutor
import time
import cv2
from scipy.spatial.transform import Rotation as R
from octo.utils.train_callbacks import supply_rng
from functools import partial

# --------------------------
# Configuración general
# --------------------------
WINDOW_SIZE = 2            # Número de frames en la secuencia (1 o 2)
proprio_info = True        # Incluir estado propioceptivo (TCP + gripper)

# Instrucciones de lenguaje y posibles modificaciones
instructions_list = [
    "drag back the brown duct tape",
    "pick up and bring back the brown duct tape",
    "pick up the brown duct tape from the blue square and place it on the yellow square"
]
goal_images_list = ["drag_back.jpg", "pick_up.jpg", "pick_place.jpg"]
task_index = 0            # Índice de la tarea a ejecutar

# Tipo de inferencia: modelo fino-ajustado o zero-shot
inference_model = "fine-tuned"      # Opciones: "fine-tuned" o "zero-shot"
inference_type = "language_instruction"  # "language_instruction", "image_conditioned" o "multimodal"

# Debug: mostrar imágenes en ventanas OpenCV
show_images = False

# Límites espaciales para movimientos (para saturar posiciones)
X_LIMS = [0.10, 0.5]
Y_LIMS = [-0.8, -0.3]
Z_LIMS = [0.05, 0.25]

# Estado del gripper (0 abierto, 1 cerrado) y registro histórico
gripper_state = 0.0
gripper_history = [gripper_state]


# --------------------------
# Nodo de captura de observaciones
# --------------------------
class ObservationCollector(Node):
    def __init__(self):
        super().__init__('observation_collector')
        # Bridge para convertir mensajes ROS/Image a numpy array
        self.bridge = CvBridge()
        self.latest_image = None
        self.current_tcp_pose = None
        # Lock para acceso concurrente a latest_image
        self.lock = threading.Lock()
        self.dg = 0.0  # Estado temporal del gripper
        self._shutdown_flag = False  # Para detener loops de espera

        # Subscríbete a los topics de imagen y pose TCP
        self.create_subscription(
            Image,
            '/camera_1/image',
            self.image_callback,
            10  # QoS
        )
        self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.tcp_pose_callback,
            10
        )

    def image_callback(self, msg):
        """
        Callback de imagen: convierte a RGB, preprocesa y guarda de forma thread-safe.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            processed = preprocess_image(cv_image)
            with self.lock:
                self.latest_image = np.array(processed)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def tcp_pose_callback(self, msg):
        """
        Callback de pose TCP: guarda x,y,z + RPY fijos + estado de gripper.
        """
        # Construir vector propioceptivo: [x,y,z,roll,pitch,yaw,gripper]
        self.current_tcp_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            0.0, -3.14, 0.0,  # Ejemplo de orientaciones fijas
            gripper_state
        ], dtype=np.float32)

    def wait_for_pose(self, target_position, tolerance=0.0001, timeout=10.0):
        """
        Espera hasta que TCP alcance target_position dentro de tolerance.
        Retorna True si lo logra, False si timeout o shutdown.
        """
        start_time = time.time()
        while not self._shutdown_flag and rclpy.ok():
            if self.current_tcp_pose is None:
                self.get_logger().warn('No pose received yet')
                time.sleep(0.1)
                continue

            # Calcular error euclidiano entre posiciones
            current_pos = self.current_tcp_pose[:3]
            error = np.linalg.norm(current_pos - np.array(target_position))
            if error < tolerance:
                self.get_logger().info(f'Position reached (error: {error:.4f}m)')
                return True

            if time.time() - start_time > timeout:
                self.get_logger().error(f'Timeout waiting for position (error: {error:.4f}m)')
                return False

            time.sleep(0.05)
        return False

    def shutdown(self):
        """Señal para terminar loops de espera."""
        self._shutdown_flag = True


# --------------------------
# Nodo para publicar comandos URScript
# --------------------------
class URScriptPublisher(Node):
    def __init__(self):
        super().__init__('urscript_publisher')
        # Publisher a topic de comandos URScript
        self.publisher = self.create_publisher(String, '/urscript_interface/script_command', 10)
        self.last_command_time = time.time()

        # Cargar plantilla de gripper desde archivo
        with open('./boilerplates/3FG15_boilerplate.txt', 'r') as f:
            self.gripper_boilerplate = f.read()

    def publish_command(self, command):
        """Publica un String con el programa URScript completo."""
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        self.last_command_time = time.time()
        self.get_logger().info('URScript command sent')

    def control_gripper(self, width):
        """
        Genera y publica comandos para abrir/cerrar el gripper.
        - width: diámetro deseado (void) para la operación.
        """
        cmd = f"tfg_release(diameter={width}, tool_index=0)"
        full = f"{self.gripper_boilerplate}\n{cmd}\nend\ntest_3FG15()"
        self.publish_command(full)


# --------------------------
# Funciones auxiliares
# --------------------------
def preprocess_image(img, angle=-3, crop_frac=1/7, output_size=(256,256)):
    """
    Igual que en el nodo anterior: rota, recorta y redimensiona la imagen RGB.
    """
    h, w = img.shape[:2]
    center = (w//2, h//2)
    rot = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(img, rot, (w, h), flags=cv2.INTER_LINEAR)
    start_x = int(w * crop_frac)
    cropped = rotated[:, start_x:]
    return cv2.resize(cropped, output_size)


def quaternion_to_rotvec(x, y, z, w):
    """Conversión de cuaternion a rotación vectorial (axis-angle)."""
    return R.from_quat([x, y, z, w]).as_rotvec()


# --------------------------
# Función principal
# --------------------------
def main(args=None):
    rclpy.init(args=args)

    # Crear nodos de captación y publicación
    collector = ObservationCollector()
    publisher = URScriptPublisher()

    # Ejecutores multihilo para procesar callbacks sin bloqueo
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(collector)
    # Lanzar spin en hilo
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Cargar modelo según configuración
    if inference_model == "fine-tuned":
        # Selección de checkpoint según task_index
        if task_index == 0:
            checkpoint = "/home/adrian/.../experiment_20250613_102430"  # ejemplo
        elif task_index == 1:
            checkpoint = "/home/adrian/.../experiment_20250613_102339"
        else:
            checkpoint = "/home/adrian/.../experiment_20250613_102449"
        model = OctoModel.load_pretrained(checkpoint)
    else:
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

    # Definir función de muestreo de acciones con RNG
    def sample_actions(model, observations, tasks, rng):
        # Obtener estadísticas de desnormalización según el modelo
        if inference_model == "fine-tuned":
            stats = model.dataset_statistics["action"]
        else:
            stats = model.dataset_statistics["berkeley_autolab_ur5"]["action"]
        obs = jax.tree_map(lambda x: x[None], observations)
        actions = model.sample_actions(obs, tasks, rng=rng, unnormalization_statistics=stats)
        return actions[0]

    policy_fn = supply_rng(partial(sample_actions, model))

    # Esperar primeras observaciones
    while rclpy.ok() and (collector.latest_image is None or collector.current_tcp_pose is None):
        time.sleep(0.1)
        print("Waiting for initial observations...")

    # Inicializar estados previos
    with collector.lock:
        previous_image = collector.latest_image.copy()
        previous_tcp = collector.current_tcp_pose

    # Crear tareas de inferencia (texto o imagen)
    if inference_type == "language_instruction":
        task = model.create_tasks(texts=[instructions_list[task_index]])
    else:
        # Cargar y preprocesar imagen objetivo
        img_path = os.path.join(os.path.dirname(__file__), "goal_images", goal_images_list[task_index])
        goal_img = cv2.imread(img_path)
        if goal_img is None:
            raise FileNotFoundError(f"Goal image not found: {img_path}")
        goal_img = preprocess_image(goal_img)
        if inference_type == "multimodal":
            task = model.create_tasks(
                texts=[instructions_list[task_index]],
                goals={"image_primary": goal_img[None]}
            )
        else:
            task = model.create_tasks(goals={"image_primary": goal_img[None]})

    total_time = 0.0
    steps = 0

    # ----------------
    # Bucle principal de control/inferencia
    # ----------------
    while rclpy.ok():
        start = time.time()
        # Copiar observaciones actuales
        with collector.lock:
            curr_img = collector.latest_image.copy()
            curr_tcp = collector.current_tcp_pose

        # Preparar secuencias de imagen y TCP según WINDOW_SIZE
        if WINDOW_SIZE == 2:
            image_seq = np.stack([previous_image, curr_img], axis=0)
            tcp_seq = np.stack([previous_tcp, curr_tcp], axis=0)
            pad = np.array([True, True])
        elif WINDOW_SIZE == 1:
            image_seq = curr_img[np.newaxis, ...]
            tcp_seq = curr_tcp[np.newaxis, ...]
            pad = np.array([True])
        else:
            raise ValueError(f"WINDOW_SIZE inválido: {WINDOW_SIZE}")

        # Construir diccionario de observación para el modelo
        obs = {"image_primary": image_seq, "timestep_pad_mask": pad}
        if proprio_info:
            obs["proprio"] = tcp_seq
        obs = jax.tree_map(lambda x: x[None], obs)

        # Muestrear acción
        stats = model.dataset_statistics["action"] if inference_model == "fine-tuned" else model.dataset_statistics["berkeley_autolab_ur5"]["action"]
        actions = model.sample_actions(obs, task, rng=jax.random.PRNGKey(0), unnormalization_statistics=stats)
        action = actions[0][0]  # Quitar dims extra

        # Descomponer acción en deltas de posición, rotación y gripper
        dx, dy, dz = action[0:3]
        delta_rot = [action[5], action[4], action[3]]
        grip_cmd = action[-1]

        # Construir nueva pose saturando límites espaciales
        pos = curr_tcp[:3].tolist()
        new_pos = [min(max(pos[i] + [dx, dy, dz][i], lim[0]), lim[1]) for i, lim in enumerate([X_LIMS, Y_LIMS, Z_LIMS])]

        # Convertir rotaciones a vectores y componer
        curr_rotvec = [-v for v in curr_tcp[3:6]]
        new_rot = (R.from_euler('xyz', delta_rot) * R.from_rotvec(curr_rotvec)).as_rotvec()

        # Comando URScript movel
        ur_cmd = (
            f"movel(p[{new_pos[0]:.6f}, {new_pos[1]:.6f}, {new_pos[2]:.6f}, "
            f"{new_rot[0]:.6f}, {new_rot[1]:.6f}, {new_rot[2]:.6f}], a=0.5, v=0.25)"
        )

        # Actualizar estado del gripper según threshold
        new_grip = 1.0 if grip_cmd >= 0.9 else 0.0
        prev_grip = 1.0 if gripper_history[-1] >= 0.9 else 0.0
        gripper_history.append(grip_cmd)

        # Log de debug
        print(f"\n--- Cycle {steps+1} ---")
        print(f"Pose: {pos}, Grip: {grip_cmd:.2f}")
        print(f"URScript: {ur_cmd}")

        # Ejecutar cambio de gripper si hay transición
        if task_index != 0:
            if new_grip and not prev_grip:
                print("Opening gripper...")
                publisher.control_gripper(width=125)
                time.sleep(3.0)
            elif not new_grip and prev_grip:
                print("Closing gripper...")
                publisher.control_gripper(width=19)
                time.sleep(3.0)

        # Mostrar imágenes si está habilitado
        if show_images:
            if WINDOW_SIZE == 2:
                seq = np.hstack([cv2.cvtColor(previous_image, cv2.COLOR_RGB2BGR), cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR)])
                cv2.imshow('Prev|Curr', seq)
                cv2.waitKey(0)
            else:
                cv2.imshow('Curr', cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)

        # Publicar movimiento URScript
        publisher.publish_command(ur_cmd)

        # Esperar a que el robot llegue (solo fine-tuned con wait_for_pose)
        if inference_model == "fine-tuned":
            if not collector.wait_for_pose(new_pos):
                collector.shutdown()
                break
        else:
            time.sleep(2.0)

        # Actualizar tiempos y contadores
        duration = time.time() - start
        total_time += duration
        steps += 1

        # Comprobar retorno a posición inicial tras 20 cycles
        if steps >= 20 and np.allclose(curr_tcp[:6], [0.15, -0.3, 0.2, 0.0, -3.14, 0.0], atol=0.025):
            print("Robot returned to start. Finishing.")
            break

        previous_image = curr_img.copy()
        previous_tcp = curr_tcp

        # Mostrar métricas parciales
        if steps > 0:
            avg = total_time / steps
            print(f"Step time: {duration:.3f}s (avg {avg:.3f}s over {steps} steps)")

    # Limpieza final
    if show_images:
        cv2.destroyAllWindows()
    executor.shutdown()
    spin_thread.join(timeout=1.0)
    collector.destroy_node()
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
