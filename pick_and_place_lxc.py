#!/usr/bin/env python
# -- coding: utf-8 --
import sys
path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
sys.path.remove(path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/home/lu/anaconda3/lib/python3.6/site-packages')
import rospy
import numpy as np
import pickle
from baselines.common import set_global_seeds
from apriltags_ros.msg import AprilTagDetectionArray
from urx import URRobot
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from math import pi
#from save_image import *
from object_detection import *
#from detect_red import detect_ball


REFERENCE_FRAME = 'frame_base'

class ur5_reach:

    def __init__(self):
        rospy.Subscriber(name='/tag_detections',
                         data_class=AprilTagDetectionArray, callback=self.on_sub, queue_size=1) # queue_size 是在订阅者接受消息不够快的时候保留的消息的数量
        #image_sub = rospy.Subscriber("/camera/rgb/image_raw",data_class=Image,callback=self.rgb_callback)
        self.target_on_base = None
        self.rbt = URRobot('192.168.31.53')

        self.gripper = Robotiq_Two_Finger_Gripper(self.rbt)
        #self.gripper.gripper_action(0)

        self.pos_init = [ 0.1, -0.3, 0.3, 0, pi, 0] 
        self.rbt.movel(self.pos_init,0.3,0.3)
        #self.gripper.gripper_action(255)
        time.sleep(2)
        self.compute_Q = False
        self.noise_eps = 0
        self.random_eps = 0
        self.use_target_net = False
        self.exploit = False
        self.has_object = True
        self.block_gripper = False #reach:true pick:false
        self.detection_target_storage = []
        #self.detection_object_storage = []
        self.gripper_state = [0, 0]
        self.pick = False
        self.object_reach = False
        #self.pick = True

        #seed = 0
        #set_global_seeds(seed)
        policy_file = "./policy_best_picknplace.pkl"

        # Load policy.
        with open(policy_file, 'rb') as f:
            self.policy = pickle.load(f)

    def detection_object_by_color(self):
        #u,v=detect_ball()
        return Real_XY_of_BOX_on_Camera(u,v,z)
    



    # rbt Transfer (rbt_base to frame_base) 
    # UR5--Base
    def on_sub(self,data):
        # target Transfer (kinect2_rgb_optical_frame to frame_base),get g 
        # Camera--Base
        detections = data.detections

        for i in range(len(detections)):
            if detections[i].id == 12:
                self.detection_target = detections[i]
                self.detection_target_storage.append([self.detection_target.pose.pose.position.x, self.detection_target.pose.pose.position.y, self.detection_target.pose.pose.position.z])
            '''
            if detections[i].id == 13:
                self.detection_object = detections[i]
                self.detection_object_storage.append([self.detection_object.pose.pose.position.x, self.detection_object.pose.pose.position.y, self.detection_object.pose.pose.position.z])
            '''

        pose_on_camera_target = self.detection_target_storage[0]
        #pose_on_camera_object = self.detection_object_storage[0]
        pose_on_camera_object = self.detection_object_by_color().ravel()


        self.target_on_base = self.camera2base(pose_on_camera_target)
        self.object_on_base = self.camera2base(pose_on_camera_object)

        pose_rbt = self.rbt.getl()

        grip_pos = [-pose_rbt[1] + 0.75, pose_rbt[0] + 0.74, pose_rbt[2] + 0.65] # grip_pos 末端位置
        target_pos = [-self.target_on_base[1] + 0.75, self.target_on_base[0] + 0.74, self.target_on_base[2] + 0.70] # 目的地位置, 目的地比夹手末端在z轴上高5cm.
        
        if self.pick == True and sum(abs(np.asarray(grip_pos)-np.asarray([-self.object_on_base[1] + 0.75, self.object_on_base[0] + 0.74, self.object_on_base[2] + 0.625]))) < 0.1:
            object_pos = grip_pos # 物体位置
            object_rel_pos = [0, 0, 0]
            gripper_state = [0.025, 0.025]
            self.object_reach = True
        
        if self.object_reach == True:
            object_pos = grip_pos # 物体位置
            object_rel_pos = [0, 0, 0]
            gripper_state = [0.025, 0.025]
        else:
            object_pos = [-self.object_on_base[1] + 0.75, self.object_on_base[0] + 0.74, self.object_on_base[2] + 0.625] # 物体位置
            object_rel_pos = [object_pos[0] - grip_pos[0], object_pos[1] - grip_pos[1], object_pos[2] - grip_pos[2]]
            gripper_state = self.gripper_state

        achieved_goal = np.squeeze(object_pos.copy())

        o = [np.concatenate([grip_pos, np.asarray(object_pos).ravel(), np.asarray(object_rel_pos).ravel(), gripper_state])] # grip_pos
        ag = [achieved_goal]  # grip_pos object_pos
        g = [target_pos]

        print(o)

        # run model get action, pos_delta (frame_base)
        # 这个action是仿真下的。1 action = 0.033 仿真观测 = 0.033真实观测。        
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
        if gripper_ctrl < 0 :
            self.pick = True
        else:
            self.pick = False

        pos_ctrl *= 0.033 #simulation action -> real world action.(1 sim action = 0.033 real action)
        gripper_ctrl *=0.7636
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion

        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl]) # action
        action, gripper_st = np.split(action, (1 * 7,))
        action = action.reshape(1, 7)
        
        pos_delta = action[:, :3]
        #pos_delta = pos_ctrl
        self.gripper_state += gripper_st
        if self.gripper_state[0] > 0.05:
            self.gripper_state = [0.05, 0.05]
        if self.gripper_state[0] < 0:
            self.gripper_state = [0, 0]

        act_on_base = pose_rbt[:3]
        her_pose = [act_on_base[0], act_on_base[1], act_on_base[2]] # 末端关于UR3基坐标系的位置


        # test on rule and get right position
        rule_pose = [0, 0, 0, 0, 0, 0]
        rule_pose[0:3] = [self.target_on_base[0], self.target_on_base[1], self.target_on_base[2]]
        rule_pose[3] = pose_rbt[3]
        rule_pose[4] = pose_rbt[4]
        rule_pose[5] = pose_rbt[5] # rx, ry, rz 固定


        # judge if the target is reached
        '''
        rule_pose_np = np.asarray(rule_pose[:3]).ravel()
        her_pose_np = np.asarray(her_pose[:3]).ravel()
        if np.matmul((rule_pose_np-her_pose_np).T, rule_pose_np-her_pose_np) < 0.02:
        '''
        if abs(her_pose[0]-rule_pose[0]) + abs(her_pose[1]-rule_pose[1]) < 0.04:
            print("reach the target")
            return

        act_on_base[0] += pos_delta[0][1]  # left right
        act_on_base[1] -= pos_delta[0][0]  # front behind
        act_on_base[2] += pos_delta[0][2]  # up down
        robot_delta = [act_on_base[0], act_on_base[1], act_on_base[2],pose_rbt[3],pose_rbt[4],pose_rbt[5]]
        
        self.rbt.movel(tpose=robot_delta, acc=0.1, vel=0.25)
        self.gripper.gripper_action(int(255 - 5100 * self.gripper_state[0]))


    def camera2base(self,pose_on_camera):
        Tm=np.array([[-0.9992,0.0082,0.0383,0.0183], [-0.0036,0.9543,-0.2987,-0.2376], [-0.0390,-0.2986,-0.9536,1.1127], [0, 0, 0, 1.0000]])
        pose_on_base=np.dot([pose_on_camera[0], pose_on_camera[1], pose_on_camera[2], 1], Tm.T)
        pose_on_base[0] = pose_on_base[0] - 0.03
        pose_on_base[1] = pose_on_base[1] * 0.98 - 0.01
        pose_on_base[2] = pose_on_base[2] * 0.94 + 0.09
        return pose_on_base


if __name__ == "__main__":
    rospy.init_node('UR5_reach') # 初始化，建立一个 python 节点
    lh = ur5_reach()
    rospy.spin() # 让程序在手动停止前一直循环