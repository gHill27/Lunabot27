#!/usr/bin/env python3

import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Pose,Twist
from nav_msgs.msg import Odometry
import message_filters
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from typing import Optional

def yaw_from_quaternion(q) -> float:
    """Convert geometry_msgs/Quaternion to yaw in radians [-pi, pi]."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

#startup node
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
        self.get_logger().info('Waiting for first odometry message...')
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)


        self.linear_speed: float = 0.0
        self.angular_speed : float = 0.5
        odom_topic = '/odom'
        cmd_vel_topic = '/cmd_vel'
        self.stop_tolerance = math.radians(2) #2 degree tolerance

        #odom
        self.start_yaw = None
        self.last_yaw = None
        self.accumulated_yaw = 0.0
        self.current_yaw = 0.0
        
        # Duration for one full 360-degree circle: 2 * pi / |angular_speed|
        self.duration_sec = (2.0 * math.pi) / abs(self.angular_speed)
        self.start_time = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def odom_callback(self, msg: Odometry):
        yaw:float = yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_yaw = yaw

        # Latch initial heading on the first /odom message
        if self.start_yaw is None:
            self.start_yaw:Optional[float] = yaw
            self.last_yaw:Optional[float] = yaw
            self.get_logger().info(
                f'Initial yaw latched at {math.degrees(self.start_yaw):.1f} deg. Starting circle!'
            )
            return #only for the first time must return 

        # Compute shortest signed delta since last odom message to handle -pi/pi wrap-around
        dyaw:float = normalize_angle(yaw - self.last_yaw) #type:ignore <--- ignore for my dumb linter
        self.accumulated_yaw += dyaw
        self.last_yaw = yaw
        return


    def timer_callback(self):
        now = self.get_clock().now()
        if self.get_clock().now().nanoseconds == 0 or self.start_yaw is None:
            return
        total_turned = abs(self.accumulated_yaw)
        remaining = (2.0 * math.pi) - total_turned

        # Check if we have completed 360 deg (within tolerance)
        if remaining <= self.stop_tolerance:
            self.cmd_pub.publish(Twist())
            self.get_logger().info(
                f'Full circle complete! Final yaw: {math.degrees(self.current_yaw):.1f} deg '
                f'(Start yaw: {math.degrees(self.start_yaw):.1f} deg, '
                f'Total turned: {math.degrees(total_turned):.1f} deg). Stopping.'
            )
            self.timer.cancel()
            raise SystemExit

        # Optional proportional slow-down in the last 15 degrees so we don't overshoot 0 deg at 10 Hz
        msg = Twist()
        slowdown_zone = math.radians(15.0)
        scale = min(1.0, max(0.25, remaining / slowdown_zone))

        msg.linear.x = self.linear_speed * scale
        msg.angular.z = self.angular_speed * scale
        self.cmd_pub.publish(msg)


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
