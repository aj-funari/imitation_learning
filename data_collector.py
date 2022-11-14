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
from helpers import resize

bridge = CvBridge()    def cmd_callback(self, msg):
        # print(msg)
        # reformat cmd_vel message to 'x-z-timestamp.jpeg' to save image 
        label = self.format_label(str(msg))
        # print("image label:", label)
        
        # change to folder directory
        directory = '/home/aj/images/avoid_walls'
        os.chdir(directory)

        # save image to folder
        cv2.imwrite(label, self.image_holder)
        self.count += 1
        print(self.count,"images saved")
move = Twist()

class data_recorder(object):

    def __init__(self):
        # Node for Subscriber/Publisher
        self.node = rospy.init_node('listener', anonymous=True)

        # Loop through at a desired rate --> 10: expect 10 times per second
        self.rate = rospy.Rate(10)
        
        # Subscribe to topics to obtain data
        self.img = rospy.Subscriber('/front/left/image_raw', Image, self.img_callback)
        self.vel = rospy.Subscriber('/jackal_velocity_controller/cmd_vel', Twist, self.cmd_callback)
    
        # Jackal camera (real world Jackal camera)
        # self.img = rospy.Subscriber('/d400/depth/image_rect_raw', Image, self.img_callback)
    
        self.image_holder = None  
        self.count = 0

    '''
    Subscribe to Jackal's front camera
        - save current snapshot to self.image --> needed to save image with label in different function
            -convert ROS image to cv2 image
            -resize image to 3 x 224 x 224 for neural network 
    '''
    def img_callback(self, image):
        # try:
            # Convert ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')  # returns array
        img = resize(cv2_img)  # resize to (224, 224, 3)
        # print(img.shape)
        self.image_holder = img

        # plt.imshow(self.image)
        # plt.show()

        # except CvBridgeError as e:
            # pass

        # self.count += 1
        # print(self.count)

    '''
    With every published veloity command ...
        1) Reformat geometry Twist message --> (x - z - timestep)
        2) Save current robot camera images to folder with linear/angular velocity as labels
    '''
    def cmd_callback(self, msg):
        # print(msg)
        # reformat cmd_vel message to 'x-z-timestamp.jpeg' to save image 
        # junhong's modification: directly getting the labels from ROS message, rather than converting it to a string
        linear_v = msg.linear.x
        angular_v = msg.angular.z
        
        label = self.format_label(str(msg))
        # print("image label:", label)
        
        # change to folder directory
        directory = '/home/aj/images/avoid_walls'
        os.chdir(directory)

        # save image to folder
        cv2.imwrite(label, self.image_holder)
        self.count += 1
        print(self.count,"images saved")

    """
    Reformat ROS cmd_vel to linear-angular-timestep-.jpeg
    - This is needed to correctly save images to folder
    """
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
