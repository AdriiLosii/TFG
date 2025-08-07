import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from ur_msgs.msg import ToolDataMsg
from cv_bridge import CvBridge
import threading
from rclpy.executors import MultiThreadedExecutor   # Ejecutor multihilo para callbacks concurrentes\import cv2
import time
import os

gripper_history = [0.0]  # Historial de estado del gripper (0 abierto, 1 cerrado)

def preprocess_image(img, angle=-3, crop_frac=1/7, output_size=(256,256)):
    """
    Aplica rotación, recorte y redimensionamiento a la imagen RGB de la cámara.
    - angle: grados a rotar (sobrescribir si es necesario invertir dirección).
    - crop_frac: fracción de ancho a descartar desde la izquierda.
    - output_size: tamaño final (ancho, alto).
    """
    h, w = img.shape[:2]
    center = (w//2, h//2)
    # Matriz de rotación centrada
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    # Recortamos una franja izquierda según crop_frac
    start_x = int(w * crop_frac)
    cropped = rotated[:, start_x:]
    # Redimensionamos a la resolución deseada
    resized = cv2.resize(cropped, output_size)
    return resized


class NpyFileGenerator(Node):
    def __init__(self):
        super().__init__('npy_file_generator')  # Nombre del nodo ROS

        # Parámetros de recogida
        task_index = 2  # Selecciona la instrucción a usar
        self.max_num_samples = 200  # Número máximo de pasos a registrar
        instructions_list = [
            "drag back the brown duct tape",
            "pick up and bring back the brown duct tape",
            "pick up the brown duct tape from the blue square and place it on the yellow square"
        ]
        self.language_instruction_string = instructions_list[task_index]

        # Determinar ruta de salida única para el .npy
        base_path = "./data"
        os.makedirs(base_path, exist_ok=True)
        episode_idx = 0
        while True:
            potential_file = os.path.join(base_path, f"episode_{episode_idx}.npy")
            if not os.path.exists(potential_file):
                self.output_file = potential_file
                break
            episode_idx += 1
        self.get_logger().info(f"Output file will be: {self.output_file}")

        # Imagen objetivo (se asignará tras recogida)
        self.goal_image = None
        # Bridge para convertir ROS Image a OpenCV
        self.bridge = CvBridge()
        # Estructura donde almacenamos todos los pasos
        self.episode_data = {'steps': []}

        # Variables para datos más recientes
        self.latest_image = None
        self.current_tcp_pose = None
        self.image_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        # Para detectar inicio de movimiento
        self.starting_pose = None
        self.has_moved = False
        self.movement_threshold = 0.005  # metros mínimo para considerar movimiento

        # Secuencias de pose para sincronización
        self.pose_seq = 0
        self.last_pose_seq = -1

        # Variables para detección de picos de corriente (gripper)
        self.prev_tool_current = 0.0
        self.peak_count = 0
        self.step_max_tool_current = 0.0

        # Subscribirse a topics necesarios
        self.create_subscription(Image, '/camera_1/image', self.image_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self.tcp_pose_callback, 10)
        self.create_subscription(ToolDataMsg, '/io_and_status_controller/tool_data', self.tool_data_callback, 50)

    def image_callback(self, msg):
        """
        Callback de imagen: convierte el mensaje ROS a OpenCV (RGB), preprocesa
        y guarda la última imagen de forma thread-safe.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            resized_image = preprocess_image(cv_image)
            with self.image_lock:
                self.latest_image = np.array(resized_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def joint_state_callback(self, msg):
        """
        Callback de estados articulares: reordena posiciones para base→wrist.
        """
        # msg.position orden: [shoulder_lift, elbow, wrist1, wrist2, wrist3, shoulder_pan]
        # Queremos: [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
        joint_state = np.array([msg.position[-1]] + list(msg.position[0:5]), dtype=np.float32)
        self.latest_joint_state = joint_state

    def tcp_pose_callback(self, msg):
        """
        Callback de pose TCP: inicializa starting_pose, detecta primer movimiento,
        y actualiza pose más reciente y secuencia.
        """
        with self.pose_lock:
            if self.starting_pose is None:
                self.starting_pose = msg.pose
                self.get_logger().info(f"Starting pose set: {self.starting_pose.position}")

            if not self.has_moved:
                cur = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
                start = np.array([self.starting_pose.position.x,
                                  self.starting_pose.position.y,
                                  self.starting_pose.position.z])
                if np.linalg.norm(cur - start) > self.movement_threshold:
                    self.has_moved = True
                    self.get_logger().info("Robot has started moving! Beginning data collection...")

            self.current_tcp_pose = msg.pose
            self.pose_seq += 1

    def tool_data_callback(self, msg: ToolDataMsg):
        """
        Callback de datos de herramienta: registra el pico de corriente
        en el intervalo de muestreo para detectar aperturas/cierres del gripper.
        """
        self.step_max_tool_current = max(self.step_max_tool_current, msg.tool_current)

    def collect_data(self):
        """
        Bucle principal de recopilación de datos:
        1. Espera datos iniciales.
        2. Espera movimiento del robot.
        3. Registra muestras hasta max_num_samples o retorno al punto inicial.
        4. Guarda los datos en un fichero .npy.
        """
        # 1) Esperar datos de imagen y pose iniciales
        while rclpy.ok() and (self.latest_image is None or self.current_tcp_pose is None):
            self.get_logger().info("Waiting for initial data...")
            time.sleep(0.1)

        # 2) Esperar primer movimiento significativo
        while rclpy.ok() and not self.has_moved:
            self.get_logger().info("Waiting for robot to start moving...", throttle_duration_sec=2)
            time.sleep(0.1)

        # Definir pose objetivo para detección de regreso
        target_pose = np.array([
            self.starting_pose.position.x,
            self.starting_pose.position.y,
            self.starting_pose.position.z,
            self.starting_pose.orientation.x,
            self.starting_pose.orientation.y,
            self.starting_pose.orientation.z
        ])
        tolerance = 0.01  # tolerancia en metros y radianes

        with self.pose_lock:
            previous_pose = self.current_tcp_pose

        # 3) Bucle de muestreo
        for i in range(self.max_num_samples):
            start_time = time.time()
            # Sincronizar con nueva pose
            while self.pose_seq == self.last_pose_seq and rclpy.ok():
                time.sleep(0.01)
            self.last_pose_seq = self.pose_seq

            # Copiar estado actual (imagen + articulaciones)
            with self.image_lock:
                image = (self.latest_image.copy()
                         if self.latest_image is not None
                         else np.zeros((256,256,3), dtype=np.uint8))
                joint_state = (self.latest_joint_state.copy()
                               if hasattr(self, 'latest_joint_state')
                               else np.zeros((6,), dtype=np.float32))

            current_tcp_pose = self.current_tcp_pose

            # Cálculo de deltas de posición con límites
            dx = np.clip(current_tcp_pose.position.x - previous_pose.position.x, -0.02, 0.02)
            dy = np.clip(current_tcp_pose.position.y - previous_pose.position.y, -0.02, 0.02)
            dz = np.clip(current_tcp_pose.position.z - previous_pose.position.z, -0.02, 0.02)
            dr = dp = dyaw = np.float32(0.0)  # rotaciones no medidas aquí

            # Detectar cambio de estado del gripper por picos de corriente
            curr = self.step_max_tool_current
            prev = self.prev_tool_current
            if prev <= 0.5 < curr:
                self.peak_count += 1
            dg = 1.0 if (self.peak_count % 2 == 1) else 0.0
            gripper_history.append(dg)
            prev_gripper = gripper_history[-2]

            # Preparar para siguiente iteración
            self.prev_tool_current = curr
            previous_pose = current_tcp_pose
            self.step_max_tool_current = 0.0

            # Construir arrays de acción y estados propioceptivos
            action = np.array([dx, dy, dz, dr, dp, dyaw, dg], dtype=np.float32)
            tcp_pose_state = np.array([
                current_tcp_pose.position.x,
                current_tcp_pose.position.y,
                current_tcp_pose.position.z,
                0.0, -3.14, 0.0  # solo ejemplo de RPY si fuera necesario
            ], dtype=np.float32)
            tcp_proprio = np.concatenate((tcp_pose_state, [prev_gripper],), dtype=np.float32)

            step_dict = {
                'observation': {'image': image, 'state': tcp_proprio},
                'action': action,
                'image_primary': None,  # se asignará después
                'language_instruction': self.language_instruction_string
            }
            self.episode_data['steps'].append(step_dict)

            # Comprobar condición de parada por retorno al punto inicial
            current_pose = np.array([
                self.current_tcp_pose.position.x,
                self.current_tcp_pose.position.y,
                self.current_tcp_pose.position.z,
                self.current_tcp_pose.orientation.x,
                self.current_tcp_pose.orientation.y,
                self.current_tcp_pose.orientation.z
            ])
            if i >= 20 and np.allclose(current_pose, target_pose, atol=tolerance):
                self.get_logger().info("Robot returned to starting position. Stopping data collection.")
                break

            # Mantener frecuencia ~5Hz
            elapsed = time.time() - start_time
            time.sleep(max(0.0, 0.2 - elapsed))

        # 4) Asignar imagen objetivo final (image_primary) en cada paso
        if self.episode_data['steps']:
            final_image = self.episode_data['steps'][-1]['observation']['image']
            for step in self.episode_data['steps']:
                step['image_primary'] = np.array(final_image, copy=True)

        # Guardar datos a disco
        np.save(self.output_file, self.episode_data)
        self.get_logger().info(f"Saved {len(self.episode_data['steps'])} samples to {self.output_file}")


def main(args=None):
    # Iniciar ROS
    rclpy.init(args=args)
    node = NpyFileGenerator()

    # Ejecutor multihilo para no bloquear callbacks
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # Ejecutar spin en hilo separado
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.collect_data()
    except KeyboardInterrupt:
        pass
    finally:
        # Cierre limpio al terminar
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()