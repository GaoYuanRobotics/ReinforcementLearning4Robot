from urx import URRobot
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper


rbt=URRobot('192.168.31.53')
robotiqgrip = Robotiq_Two_Finger_Gripper(rbt)
pos_init = [ 0.1, -0.4, 0.35, 2, 0, 0]
robotiqgrip.open_gripper()
rbt.movel(pos_init,0.2,0.2)
print("Robot is moving...")
rbt.close()
