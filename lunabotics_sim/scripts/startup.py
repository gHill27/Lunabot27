#!/usr/bin/env python3

import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Pose,Twist
import message_filters
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

#Fiducial tracker node
class Startup(Node):
    def __init__(self):
        super().__init__('startupNode')
        if not self.get_parameter('use_sim_time').value:
            topic_names = [name for name, _ in self.get_topic_names_and_types()]
            if '/clock' in topic_names:
                self.get_logger().info('Detected /clock topic; enabling use_sim_time automatically.')
                self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.using_sim_time = self.get_parameter('use_sim_time').value
        clock_type = 'SIMULATION time (/clock)' if self.using_sim_time else 'SYSTEM wall time'
        self.get_logger().info(f'Initializing CircleDriver using {clock_type} at 10 Hz.')

        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 0.5)
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed : float = self.get_parameter('angular_speed').value
        
        # Duration for one full 360-degree circle: 2 * pi / |angular_speed|
        self.duration_sec = (2.0 * math.pi) / abs(self.angular_speed)
        self.start_time = None
        self.publsiher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        self.driveinCircle()

    def driveinCircle(self):
        twist_msg = Twist()
        twist_msg.linear.x = 0.2
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0

        # Set angular velocity (roll, pitch, yaw)
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.3

def timer_callback(self):
        now = self.get_clock().now()

        # When use_sim_time is True, clock returns 0 until the first /clock message arrives.
        # When use_sim_time is False, now.nanoseconds is immediately > 0 (Unix epoch).
        if now.nanoseconds == 0:
            self.get_logger().info('Waiting for /clock to publish...', throttle_duration_sec=2.0)
            return

        # Latch start time on the first valid tick
        if self.start_time is None:
            self.start_time = now
            self.get_logger().info(
                f'Starting circle: v={self.linear_speed} m/s, w={self.angular_speed} rad/s '
                f'for {self.duration_sec:.2f}s.'
            )

        elapsed = (now - self.start_time).nanoseconds / 1e9

        # Handle Gazebo simulation resets (where sim time jumps backward)
        if elapsed < 0.0:
            self.get_logger().warn('Clock jumped backward (sim reset detected). Resetting start time.')
            self.start_time = now
            elapsed = 0.0

        msg = Twist()
        if elapsed < self.duration_sec:
            msg.linear.x = float(self.linear_speed)
            msg.angular.z = float(self.angular_speed)
            self.cmd_pub.publish(msg)
        else:
            # Publish zero velocity to stop the robot before exiting
            self.cmd_pub.publish(Twist())
            self.get_logger().info('Circle complete! Stopping robot and shutting down node.')
            self.timer.cancel()
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = Startup()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
