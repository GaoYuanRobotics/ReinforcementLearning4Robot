#!/usr/bin/env python
# -- coding: utf-8 --
import sys
path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
sys.path.remove(path)
import cv2
sys.path.append(path)
import rospy
import numpy as np
import pickle

from baselines.common import set_global_seeds

from urx import URRobot
from math import pi
from color_detection_2 import ColorDetection
#from img import *

from time import gmtime, strftime
from std_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rospy_tutorials.msg import Floats

class ur5_reach:

    def __init__(self):
        self.target_on_base = None
        self.rgb_image = None
        self.depth_image = None
        self.rbt = URRobot('192.168.31.53')
        self.compute_Q = False
        self.noise_eps = 0
        self.random_eps = 0
        self.use_target_net = False
        self.exploit = False
        self.block_gripper = True #reach:true pick:false
        #self.detection_storage = []
        seed = 0
        set_global_seeds(seed)
        policy_file = "/home/lu/liboyao/policy_best_reach.pkl"
        # Load policy.
        with open(policy_file, 'rb') as f:
            self.policy = pickle.load(f)

    def rgb_callback(self, imgmessage):
        rgb_bridge = CvBridge()
        self.rgb_image = rgb_bridge.imgmsg_to_cv2(imgmessage, "bgr8")
        detect=ColorDetection(self.rgb_image)
        msg_to_send=Floats()
        msg_to_send.data=detect.image_uv(2)
        mypub= rospy.Publisher("uvtopic", Floats,queue_size=1)
        mypub.publish(msg_to_send)

    def depth_callback(self, depthmessage):
        dep_bridge = CvBridge()
        self.depth_image = dep_bridge.imgmsg_to_cv2(depthmessage,"passthrough")
        #cv2.imwrite('/home/lu/liboyao/depth.png',depth_image)



    def on_sub(self):
        z = self.depth_image[v][u] + 100
        u /= 1.0
        v /= 1.0
        z /= 1000.0
        print(u, v, z)
        M_in=np.array([[570.342,0,319.5],[0,570.342,239.5],[0,0,1]])
        uv1 = np.array([[u,v,1]]) * z #uv1=[u,v,1] *z
        xy1 = np.dot(np.linalg.inv(M_in),uv1.T) #xy1=[x,y,1] real world xy
        pose_on_camera = xy1
        #detection=ColorDetection(rgbimage, depthimage) # build a 'method'
        #pose_on_camera_x,pose_on_camera_y,pose_on_camera_z=detection.image2camera(2) # 2 is orange
        #pose_on_camera=[pose_on_camera_x,pose_on_camera_y,pose_on_camera_z]
        self.target_on_base = self.camera2base(pose_on_camera)
        #print(pose_on_camera)
        #print(self.target_on_base)
        '''detections = data.detections

        for i in range(len(detections)):
            if detections[i].id == 12:
                self.detection_target = detections[i]
                self.detection_storage.append([self.detection_target.pose.pose.position.x,self.detection_target.pose.pose.position.y,self.detection_target.pose.pose.position.z])
                break

        pose_on_camera = self.detection_storage[0]

        self.target_on_base = self.camera2base(pose_on_camera)'''


        pose_rbt = self.rbt.getl()
        
        o = [-pose_rbt[1] + 0.75, pose_rbt[0] + 0.74, pose_rbt[2] + 0.65] # grip_pos 末端位置
        ag = o # grip_pos object_pos
        g = [-self.target_on_base[1] + 0.76, self.target_on_base[0] + 0.74, self.target_on_base[2] + 0.64] # 目标位置

        action =self.policy.get_actions(
                o, ag, g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

        if self.compute_Q:
            u, Q = action
        else:
            u = action

        if u.ndim == 1:
            # The non-batched case should still have a reasonable shape.
            u = u.reshape(1, -1)
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.033
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion

        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl]) # action
        action, _ = np.split(action, (1 * 7,))
        action = action.reshape(1, 7)
        pos_delta = action[:, :3]

        act_on_base = pose_rbt[:3]
        her_pose = [act_on_base[0], act_on_base[1], act_on_base[2]] # 末端关于UR3基坐标系的位置


        # test on rule and get right position
        rule_pose = [0, 0, 0, 0, 0, 0]
        rule_pose[0:3] = [self.target_on_base[0], self.target_on_base[1], self.target_on_base[2]]
        rule_pose[3] = pose_rbt[3]
        rule_pose[4] = pose_rbt[4]
        rule_pose[5] = pose_rbt[5] # rx, ry, rz 固定


        # judge if the target is reached
        if sum(np.square(np.asarray(her_pose) - np.asarray([rule_pose[0],rule_pose[1],rule_pose[2]]))) < 0.0005:
            print("reach the target")
            return
        
        act_on_base[0] += pos_delta[0][1]  # left right
        act_on_base[1] -= pos_delta[0][0]  # front behind
        act_on_base[2] += pos_delta[0][2]  # up down
        robot_delta = [act_on_base[0], act_on_base[1], act_on_base[2],pose_rbt[3],pose_rbt[4],pose_rbt[5]]
        self.rbt.movel(tpose=robot_delta, acc=0.1, vel=0.15)
  
        print('----------pos_delta----------')
        print(pos_delta)
        print('----------robot_pose----------')
        print([act_on_base[0], act_on_base[1], act_on_base[2]])
        print('----------rule_pose----------')
        print(rule_pose)

    def camera2base(self,pose_on_camera):
        Tm=np.array([[-0.9992,0.0082,0.0383,0.0183], [-0.0036,0.9543,-0.2987,-0.2376], [-0.0390,-0.2986,-0.9536,1.1127], [0, 0, 0, 1.0000]])
        #pose_on_base=np.dot([pose_on_camera.x,pose_on_camera.y,pose_on_camera.z,1],Tm.T)
        pose_on_base=np.dot([pose_on_camera[0], pose_on_camera[1], pose_on_camera[2], 1], Tm.T)
        pose_on_base[0] = pose_on_base[0] - 0.03
        pose_on_base[1] = pose_on_base[1] * 0.98 - 0.01
        pose_on_base[2] = pose_on_base[2] * 0.94 + 0.09
        return pose_on_base
    
    def rbt_init(self,pos_init): 
        self.rbt.movel(pos_init,0.1,0.15)


if __name__ == "__main__":
    #rospy.init_node('UR5_reach') # 初始化，建立一个 python 节点
    lh = ur5_reach()
    rospy.Subscriber("/camera/rgb/image_raw",Image, lh.rgb_callback)
    rospy.Subscriber("/camera/depth_registered/image",Image, lh.depth_callback)
    rospy.Subscriber("/callback_u", Int16, lh.receive_u)
    rospy.Subscriber("/callback_v", Int16, lh.receive_v)
    #pos_init = [ 0, -0.3, 0.3, 0, pi, 0]
    pos_init = [ 0.1, -0.4, 0.35, 2, 0, 0]
    lh.rbt_init(pos_init)
    while True:
        lh.on_sub()
    #rospy.spin() # 让程序在手动停止前一直循环

