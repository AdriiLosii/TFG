#!/usr/bin/env python3
import os
# Desactivar paralelismo en tokenizers para evitar warnings en multi-hilo
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import numpy as np
import time
import threading

class URScriptPublisher(Node):
    """
    Nodo ROS que publica comandos URScript y espera hasta alcanzar posiciones.
    """
    def __init__(self):
        super().__init__('urscript_publisher')
        # Publisher para enviar comandos URScript al controlador
        self.publisher = self.create_publisher(String, '/urscript_interface/script_command', 10)
        # Almacena la última pose TCP recibida
        self.current_pose = None
        # Subscriber para recibir pose TCP en topic adecuado
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/tcp_pose_broadcaster/pose',
            self.pose_callback,
            10
        )
        # Bandera para indicar cierre de loops de espera
        self._shutdown_flag = False
        
        # Cargar plantilla del gripper desde archivo externo
        with open('./boilerplates/3fg15_boilerplate.txt', 'r') as f:
            self.gripper_boilerplate = f.read()

    def pose_callback(self, msg):
        """Callback que almacena la pose TCP si no estamos cerrando."""
        if not self._shutdown_flag:
            self.current_pose = msg.pose

    def publish_command(self, command):
        """Publica un comando URScript como mensaje String."""
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        self.get_logger().info('URScript command sent')

    def control_gripper(self, width):
        """
        Construye y publica el programa URScript para operar el gripper.
        - width: diámetro deseado para abrir o cerrar.
        """
        # Comando específico para el gripper
        command = f"tfg_release(diameter={width}, tool_index=0)"
        # Combinar plantilla con comando y clausura de programa
        full_program = f"{self.gripper_boilerplate}\n{command}\nend\ntest_3FG15()"
        self.publish_command(full_program)

    def wait_for_pose(self, target_position, tolerance=0.005, timeout=10.0):
        """
        Espera hasta que la pose TCP alcance target_position dentro de una tolerancia.
        Args:
            target_position: lista [x, y, z] en metros.
            tolerance: error máximo aceptable en metros.
            timeout: tiempo máximo de espera en segundos.
        Retorna True si alcanza, False si falla o timeout.
        """
        start_time = time.time()
        # Bucle de espera activo hasta shutdown o fin de ROS
        while not self._shutdown_flag and rclpy.ok():
            # Si aún no hay pose, avisar y continuar esperando
            if self.current_pose is None:
                self.get_logger().warn('No pose received yet')
                time.sleep(0.1)
                continue

            # Extraer coordenadas actuales
            current_pos = [
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ]
            # Calcular error euclidiano
            error = np.linalg.norm(np.array(current_pos) - np.array(target_position))

            # Si está dentro de tolerancia, informar y salir
            if error < tolerance:
                self.get_logger().info(f'Position reached with error: {error:.4f}m')
                return True

            # Si supera tiempo de espera, informar error y salir
            if time.time() - start_time > timeout:
                self.get_logger().error(f'Timeout waiting for position (current error: {error:.4f}m)')
                return False
            
            time.sleep(0.05)
        return False

    def shutdown(self):
        """Señala que el nodo debe terminar sus loops de espera."""
        self._shutdown_flag = True


def main(args=None):
    # Inicializar ROS
    rclpy.init(args=args)
    # Crear instancia del nodo publicador
    publisher = URScriptPublisher()

    # Spin en un hilo separado para procesar callbacks sin bloquear
    spin_thread = threading.Thread(target=rclpy.spin, args=(publisher,), daemon=True)
    spin_thread.start()

    try:
        # Esperar a la primera pose antes de enviar comandos
        while publisher.current_pose is None and not publisher._shutdown_flag and rclpy.ok():
            publisher.get_logger().info('Waiting for initial pose...')
            time.sleep(0.5)

        # Construir y publicar comando de movimiento a posición inicial
        start_pose = [0.15, -0.30, 0.20, 0.0, -3.14, 0.0]
        tcp_position_command = f"movel(p{start_pose}, a=1.2, v=0.25)"
        print(f"\nURScript Command: {tcp_position_command}")    
        publisher.publish_command(tcp_position_command)
        
        # Esperar a que el robot alcance la posición deseada
        if not publisher.wait_for_pose(start_pose[:3]):
            publisher.get_logger().error('Failed to reach target position')
            publisher.shutdown()
            return

        # Mover el gripper tras posicionarse
        print("Moving gripper...")
        publisher.control_gripper(width=19)
        time.sleep(1.0)

    except KeyboardInterrupt:
        # Manejo de interrupción manual (Ctrl+C)
        publisher.get_logger().info('Shutting down due to keyboard interrupt')
    finally:
        # Cierre limpio: señalizar shutdown y terminar nodo
        publisher.shutdown()
        rclpy.shutdown()
        if spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
        publisher.destroy_node()


if __name__ == '__main__':
    main()