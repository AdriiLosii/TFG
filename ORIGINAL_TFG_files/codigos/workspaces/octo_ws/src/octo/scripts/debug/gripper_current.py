#!/usr/bin/env python3
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import rclpy
from rclpy.node import Node
from ur_msgs.msg import ToolDataMsg
import time
import threading

import matplotlib.pyplot as plt  # Para la gráfica

class ToolCurrentLogger(Node):
    def __init__(self):
        super().__init__('tool_current_logger')
        # Suscripción a tool_data
        self.create_subscription(
            ToolDataMsg,
            '/io_and_status_controller/tool_data',
            self._on_tool_data,
            10
        )
        # Listas para almacenar tiempos y corrientes
        self.times = []
        self.currents = []
        self._start_time = time.time()
        self._shutdown = False

    def _on_tool_data(self, msg: ToolDataMsg):
        if self._shutdown:
            return
        t = time.time() - self._start_time
        self.times.append(t)
        self.currents.append(msg.tool_current)
        # opcional: hacer debug
        # self.get_logger().debug(f"[{t:.2f}s] tool_current = {msg.tool_current:.3f}A")

    def shutdown(self):
        self._shutdown = True

def main(args=None):
    rclpy.init(args=args)
    node = ToolCurrentLogger()

    # Lanzamos el spin en segundo plano
    spin_thr = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thr.start()

    # Recogemos datos durante 10 segundos
    time.sleep(10.0)
    node.get_logger().info('10 s transcurridos. Deteniendo suscripción y preparando gráfica.')

    # Paramos el callback y el spin
    node.shutdown()
    rclpy.shutdown()
    spin_thr.join(timeout=1.0)

    # -- Plot --
    plt.figure()
    plt.plot(node.times, node.currents)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Corriente de herramienta (A)')
    plt.title('Tool Current vs. Time (10 s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
