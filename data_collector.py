#!/usr/bin/env python

import os
import cv2
import csv
import torch
import rospy
from datetime import datetime
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image

bridge = CvBridge()
move = Twist()

class data_recorder(object):

    def __init__(self):
        # self.data = None
        self.image = None
        self.tensor_x_z_actions = []
        self.count = 0
        # Node for Subscriber/Publisher
        self.node = rospy.init_node('listener', anonymous=True)
        self.img = rospy.Subscriber('/front/left/image_raw', Image, self.img_callback)
        # self.img = rospy.Subscriber('/d400/depth/image_rect_raw', Image, self.img_callback)
        self.vel = rospy.Subscriber('/jackal_velocity_controller/cmd_vel', Twist, self.cmd_callback)
        self.rate = rospy.Rate(10)
    
    '''
    Subscribe to Jackal's front camera 
        -convert ROS image to cv2 image
        -save current camera image/view to class variable 
    '''
    def img_callback(self, image):
        # try:
            # Convert ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')  # returns array
        self.image = cv2_img

        # check image is being received
        # plt.imshow(self.image)
        # plt.show()

        # except CvBridgeError as e:
            # pass

    '''
    With every published veloity command ...
        1) Reformat geometry Twist message --> (x - z - timestep)
        2) Save current robot camera images to folder linear/angular velocity as labels
    '''
    def cmd_callback(self, msg):
        # reformat cmd_vel message to 'x-z-timestamp.jpeg' to save image 
        label = self.format_label(str(msg))
        print("image label:", label)
        
        # change to folder directory
        directory = '/home/aj/images/avoid_walls'
        os.chdir(directory)

        # save image to folder
        cv2.imwrite(label, self.image)
        self.count += 1
        # print(self.count,"images saved")

    def format_label(self, string):
        msg = string.split()
        x = msg[2]
        z = msg[13]
        msg = x + '-' + z
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = msg + '-' + current_time + '-' + '.jpeg'
        return(msg)

if __name__ =='__main__':
    print("I am in main!")
    data_recorder()
    rospy.spin()
