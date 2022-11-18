#!/usr/bin/env python

import cv2
import torch
import rospy
import numpy
from geometry_msgs.msg import Twist # message type for cmd_vel
from sensor_msgs.msg import Image # message type for image
from cv_bridge import CvBridge, CvBridgeError
from model import CNN
from helpers import resize

bridge = CvBridge()
move = Twist()

### LOAD MODEL
model = CNN(image_channels=3, num_classes=2)
PATH = '/home/aj/catkin_ws/src/imitation_learning/models/loss_0.4217402935028076.pt'
model.load_state_dict(torch.load(PATH))
model.eval()

class publish_action(object):

    def __init__(self):
        # Node for Subscriber/Publisher
        self.node = rospy.init_node('talker', anonymous=True)
        self.img = rospy.Subscriber('/front/left/image_raw', Image, self.left_img_callback)
        # self.img_left = rospy.Subscriber('/d400/depth/image_rect_raw', Image, self.left_img_callback)
        self.pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10) # definging the publisher by topic, message type
        self.rate = rospy.Rate(10)

        self.actions = []
        self.linear = None
        self.angular = None

        # self.data = None
        # self.left_image = None
        # self.right_image = None
        # self.count = 0

    def left_img_callback(self, image):
        
        # try:
        cv2_img = bridge.imgmsg_to_cv2(image)  # returns array
        # img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
        # print("Image received and converted to RGB!")

        # resize image 
        img_resize = resize(cv2_img)  # resize image to (3 x 224 x 224)
        img_float32 = numpy.array(img_resize, dtype=numpy.float32)
        img_tensor = torch.from_numpy(img_float32)
        image = img_tensor.reshape(1, 3, 224, 224)
        image = image / 255.0
        
        # move images to GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # image = image.to(device)
        
        # feed image through neural network
        tensor_out = model(image)
        self.actions.append(tensor_out)
        # print("tensor out:", tensor_out)
        # print("linear:", tensor_out[0][0])
        # print("angular:", tensor_out[0][1])

        self.linear = tensor_out[0][0]
        self.angular = tensor_out[0][1]

        # except CvBridgeError as e:
        #     pass

    def publishMethod(self):    
        i = 0
        tmp = 1
        while not rospy.is_shutdown():
            # handle delay between subscriber and publisher
            if len(self.actions) == 0:
                pass
            else:
                if len(self.actions) >= tmp:  # publish actions only when action is sent from neural network output
                    # print("x-z actions:", self.tensor_x_z_actions[i])
                    # x = move.linear.x = self.actions[i][0][0]
                    # z = move.angular.z = self.actions[i][0][1]
                    # move.linear.x = self.actions[i][0][0]
                    # move.angular.z = self.actions[i][0][1]
                    x = move.linear.x = self.linear
                    z = move.angular.z = self.angular
                    print(x, z)
                    rospy.loginfo("Data is being sent") 
                    self.pub.publish(move)
                    self.rate.sleep()
                    i += 1
                    tmp += 1

if __name__ =='__main__':
    pub = publish_action()
    pub.publishMethod()
    rospy.spin()