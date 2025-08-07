#!/usr/bin/env python3
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import urx


class URScriptPublisher(Node):
    def __init__(self):
        super().__init__('urscript_publisher')
        self.publisher = self.create_publisher(String, '/urscript_interface/script_command', 10)

    def publish_command(self, command):
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        self.get_logger().info(f'URScript command sent')


def main(args=None):
    rclpy.init(args=args)
    publisher = URScriptPublisher()
    rob = urx.Robot("169.254.128.101")

    # Wait for publisher to initialize
    time.sleep(1.0)

    # Move to start position
    tcp_position_command = "movel(p[0.10, -0.40, 0.40, 0.0, -3.14, 0.0], a=1.2, v=0.25)"
    print(f"\nURScript Command: {tcp_position_command}")
    publisher.publish_command(tcp_position_command)



    # Load gripper boilerplate from file
    with open('./boilerplates/3FG15_boilerplate.txt', 'r') as f:
        gripper_boilerplate = f.read()

    """Control the gripper with specified parameters"""
    close_grip_command = f"tfg_release(diameter=35.0, tool_index=0)"
    full_program = f"{gripper_boilerplate}\n{close_grip_command}\nend\nProgram()"

    # Open gripper
    print("\nOpening gripper...")
    rob.send_program(full_program)
    time.sleep(1.0)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()