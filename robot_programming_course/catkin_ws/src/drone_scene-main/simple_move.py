#!/usr/bin/env python

import time

import rospy
from geometry_msgs.msg import Twist

from hector_uav_msgs.srv import EnableMotors


class SimpleMover():

    def __init__(self):
        rospy.init_node('simple_mover', anonymous=True)
        self.rate = rospy.Rate(30)

        # (1) Initialize publisher to 'cmd_vel' topic
        # # TODO Uncomment and Modify following code to 
        # self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        rospy.on_shutdown(self.shutdown)

    def enable_motors(self):
        # (2) Call here the ros-service 'enable_motors'
        # TODO write code here

    def take_off(self):
        self.enable_motors()
        start_time = time.time()
        end_time = start_time + 3

        # (3) Set the linear velocity by z axis during 3 seconds
        # TODO write code here

    def spin(self):

        self.take_off()

        while not rospy.is_shutdown():
            
            # (4) Set the linera velocity by x axis is equal to 0.5 m/sec
            # TODO write code here

            self.rate.sleep()

    def shutdown(self):
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


if __name__=="__main__":
    simple_mover = SimpleMover()
    simple_mover.spin()