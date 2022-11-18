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

bridge = CvBridge()    
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

        # list for csv files
        self.image_labels = []
        # self.linear_v = []
        # self.angular_v = []

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
        self.image_holder = img

        # plt.imshow(self.image)
        # plt.show()

        # except CvBridgeError as e:
            # pass

    '''
    With every published veloity command ...
        1) Reformat geometry Twist message --> (x - z - timestep)
        2) Save current robot camera images to folder with linear/angular velocity as labels
    '''
    def cmd_callback(self, msg):
        self.count += 1
        print(f"cmd_callback called {self.count} times")
    
        # junhong's modification: directly getting the labels from ROS message, rather than converting it to a string
        linear_v = msg.linear.x
        angular_v = msg.angular.z

        # reformat cmd_vel message to 'x-z-timestamp.jpeg' to save image
        label = self.format_label(linear_v, angular_v)

        # change folder directory
        directory = '/home/aj/catkin_ws/src/imitation_learning/images'
        # directory = '/home/aj/images/avoid_walls'
        os.chdir(directory)

        # save image to folder
        cv2.imwrite(label, self.image_holder)
        self.image_labels.append(label)  # save label
        
        
        """
        Current issue:
        - When saving images to folder, the number of images being saved does not match the
          number of times the callback function is being called. Every time the callback function
          is called, I am appending the linear, angular, and image label to a list. This list is 
          then used to create my csv file. Since the amount of images being saved does not match
          the number of callbacks, the data in the csv file is much greater than the actual images
          in the foler. I tried to implement two checks to see if I can only append data to list 
          for csv file when the images are being saved. I tried to check the length of the foler
          and only append data when the size of the folder increases. My code does not work. 
          Attempts are commented out below. Instead I will save images to the folder and directly
          build the csv file from the labels in the folder. 
        """


    """
    Reformat ROS Twist message to linear-angular-timestep-.jpeg
    - This is needed to save images to folder
    """

    def format_label(self, linear_v, angular_v): 
        x = str(linear_v)
        z = str(angular_v)
        msg = x + '-' + z
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = msg + '-' + current_time + '-' + '.jpeg'
        return(msg)  # e.g. 1.9542900323867798-0.03690094836056232-13:30:34-.jpeg

if __name__ =='__main__':
    print("I am in main!")
    D = data_recorder()
    rospy.spin()

    """
    On exit of ROS node, create labels.csv file --> annotations file for customized dataset
    - save image label, linear velocity, angular velocity
    """

    directory = '/home/aj/catkin_ws/src/imitation_learning/images'
    print("Number of images saved:", len(os.listdir(directory)))

    print("Number of labels saved:", len(D.image_labels))


    # f = open("/home/aj/catkin_ws/src/imitation_learning/labels.csv", "w")
    # for i in range(len(D.linear_v)):
    #     f.write(f"{D.image_labels[i]}, {D.linear_v[i]}, {D.angular_v[i]} \n")
    # f.close()
