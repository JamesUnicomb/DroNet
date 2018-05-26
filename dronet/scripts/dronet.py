#!/usr/bin/env python

from DroNet import DroNet

import rospy

from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge, CvBridgeError

from math import *


class DroneCommander:
    def __init__(self,
                 max_velocity   = 0.125,
                 alpha          = 0.5,
                 beta           = 0.5,
                 slip_factor    = 0.0014,
                 steering_ratio = 15.3,
                 wheel_base     = 2.67,
                 image_topic    = '/bebop/image_raw'):

        self.max_velocity   = max_velocity
        self.alpha          = alpha
        self.beta           = beta
        self.slip_factor    = slip_factor
        self.steering_ratio = steering_ratio
        self.wheel_base     = wheel_base

        self.current_velocity   = 0.0
        self.current_turn_angle = 0.0

        print 'Loading DroNet... '
        self.dn = DroNet(is_training = False)
        print 'DroNet loaded'

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
        self.twist_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=5)


    def image_callback(self,
                       image):

        try:
            cv_image    = self.bridge.imgmsg_to_cv2(image, "bgr8")
            gray_image  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)[4:-4,80:-80]
            input_image = cv2.resize(gray_image, (0,0), fx = 1.0 / 3.0, fy = 1.0 / 3.0)

        except CvBridgeError as e:
            print(e)

        turn_angle, collision_probability = self.dn(input_image)

        self.current_velocity   = self.alpha * self.current_velocity + (1 - self.alpha) * (1.0 - collision_probability) * self.max_velocity
        self.current_turn_angle = self.beta * self.current_turn_angle + (1 - self.beta) * pi / 2.0 * turn_angle

        angular_velocity = 0.0 #self.current_velocity * self.current_turn_angle / (self.steering_ratio * self.wheel_base * (1 + self.slip_factor * self.current_velocity * self.current_velocity))

        t = Twist()
        t.linear.x  = self.current_velocity
        t.angular.z = angular_velocity

        self.twist_pub.publish(t)




def main():
    rospy.init_node('drone_commander', anonymous=True)
    dc = DroneCommander()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    rospy.spin()


if __name__=='__main__':
    main()
