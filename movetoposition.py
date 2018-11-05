from urx import URRobot
rbt=URRobot('192.168.31.53')
import numpy as np
def camera2base(pose_on_camera):
    Tm=np.array([[-0.9992,0.0082,0.0383,0.0183], [-0.0036,0.9543,-0.2987,-0.2376], [-0.0390,-0.2986,-0.9536,1.1127], [0, 0, 0, 1.0000]])
    #pose_on_base=np.dot([pose_on_camera.x,pose_on_camera.y,pose_on_camera.z,1],Tm.T)
    pose_on_base=np.dot([pose_on_camera[0], pose_on_camera[1], pose_on_camera[2], 1], Tm.T)
    pose_on_base[0] = pose_on_base[0] - 0.03
    pose_on_base[1] = pose_on_base[1] * 0.98 - 0.01
    pose_on_base[2] = pose_on_base[2] * 0.94 + 0.09
    return pose_on_base


pose_on_camera = [0.12036043,-0.04154134, 1.215]
pose_on_base = camera2base(pose_on_camera)
destination = [pose_on_base[0], pose_on_base[1], pose_on_base[2], 3.14, 0, 0]
print(destination)
rbt.movel(destination,0.1,0.1)
rbt.close()
