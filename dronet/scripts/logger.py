#!/usr/bin/env python
import os
from datetime import datetime

import rospy
from rospkg import RosPack
import message_filters

from std_msgs.msg import Float32, UInt8
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image, CameraInfo, NavSatFix

class logger:
    def __init__(self,
                 camera_image_topic = '/usb_cam/image_raw',
                 camera_info_topic  = '/usb_cam/camera_info'):

        self.pkgdir = RosPack().get_path('dronet')
        self.logdir = os.path.join(self.pkgdir,'log')

        self.fname = 'log%d.txt'%(len(os.listdir(self.logdir)))
        self.f = open(os.path.join(self.logdir, self.fname), 'w')
        self.f.write('time,' + 
                     'linear.velocity.x,' +
                     'linear.velocity.y,' +
                     'linear.velocity.z,' +
                     'angular.velocity.x,' +
                     'angular.velocity.y,' +
                     'angular.velocity.z,' +
                     'gps.latitude,' +
                     'gps.longitude,' + 
                     'gps.height,' + 
                     'gps.health')
        self.f.close()

        self.linear_velocity_sub  = message_filters.Subscriber('/dji_sdk/velocity', Vector3Stamped)
        self.angular_velocity_sub = message_filters.Subscriber('/dji_sdk/angular_velocity_fused', Vector3Stamped)
        self.gps_position_sub     = message_filters.Subscriber('/dji_sdk/gps_position', NavSatFix)

        self.gps_sub              = rospy.Subscriber('/dji_sdk/gps_health', UInt8, self.gps_health_cb)

        self.logger_status        = rospy.Publisher('/dronet/logging_status', UInt8, queue_size=5)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.linear_velocity_sub,
                                                               self.angular_velocity_sub,
                                                               self.gps_position_sub], 10, 1.0)

        self.ts.registerCallback(self.fc_cb)



    def fc_cb(self,
              lin_vel, ang_vel, gps_pos):
        now = rospy.get_time()

        logstring = '\n%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d' % (now,
                                                       lin_vel.vector.x,
                                                       lin_vel.vector.y,
                                                       lin_vel.vector.z,
                                                       ang_vel.vector.x,
                                                       ang_vel.vector.y,
                                                       ang_vel.vector.z,
                                                       gps_pos.latitude,
                                                       gps_pos.longitude,
                                                       gps_pos.altitude,
                                                       self.gps_health)

        self.f = open(os.path.join(self.logdir, self.fname), 'a')
        self.f.write(logstring)
        self.f.close()

        try:
            self.logger_status.publish(UInt8(1))
        except ROSException as e:
            print e


    def gps_health_cb(self,
                      data):
       self.gps_health = data.data



def main():
    rospy.init_node('dronet_data_logger')
    lg = logger()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    rospy.spin()


if __name__=='__main__':
    main()
