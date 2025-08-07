import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os


class SaveImageNode(Node):
    """
    Nodo ROS que se suscribe al tópico de imagen de la cámara
    y guarda cada mensaje de imagen en disco como archivo JPEG.
    """
    def __init__(self):
        super().__init__('save_image_node')
        # Subscribirse al tópico de imágenes
        self.subscription = self.create_subscription(
            Image,
            '/camera_1/image',
            self.save_image_callback,
            10  # Tamaño de la cola
        )
        # Bridge para convertir entre sensor_msgs/Image y OpenCV
        self.bridge = CvBridge()
        self.get_logger().info("Subscribed to /camera_1/image. Waiting for images...")

    def save_image_callback(self, msg):
        """
        Callback que recibe el mensaje ROS Image, lo convierte a formato OpenCV (RGB),
        y lo guarda en el directorio ./goal_images con nombre configurado.
        """
        try:
            # Convertir ROS Image a numpy array BGR o RGB
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Directorio y nombres de archivo posibles
        save_dir = "./goal_images"
        file_name_list = [
            "drag_back.jpg",
            "pick_up.jpg",
            "pick_place.jpg"
        ]
        # Escoger índice según el caso de uso (0,1 o 2)
        file_name = file_name_list[2]
        save_path = os.path.join(save_dir, file_name)

        # Crear directorio si no existe
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        try:
            # Escribir la imagen en disco (formato JPEG)
            cv2.imwrite(save_path, cv_image)
            self.get_logger().info(f"Processed image saved to {save_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")


def main(args=None):
    # Inicializar el cliente ROS 2
    rclpy.init(args=args)
    # Crear instacia del nodo
    node = SaveImageNode()
    try:
        # Ejecutar loop de callbacks hasta interrupción
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Manejo de Ctrl+C
        pass
    finally:
        # Destruir nodo y cerrar ROS
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()