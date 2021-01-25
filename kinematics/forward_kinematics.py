'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for different joints
'''

# add PYTHONPATH
import os
import sys
import numpy as np
from math import sin, cos, sqrt
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from numpy.matlib import matrix, identity

from angle_interpolation import AngleInterpolationAgent


class ForwardKinematicsAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
			            'LArm': ['LShoulderPitch', 'LShoulderRoll' , 'LElbowYaw' , 'LElbowRoll'],
                        'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'RAnkleRoll'],
                        'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'LAnkleRoll'],
                        'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']    	
                       }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        T = identity(4)
        # YOUR CODE HERE

        s = sin(joint_angle)
        c = cos(joint_angle)

        offsets = {
        'HeadYaw': [0.0, 0.0, 100],
        'HeadPitch': [0.0, 10, 0.0],
        'LShoulderPitch': [0.0, 60, 70],
        'LShoulderRoll': [0.0, 0.0, 0.0],
        'LElbowYaw': [50, 20, 0.0],
        'LElbowRoll': [0.0, 0.0, 0.0],
        'RShoulderPitch': [0.0, -60, 40],
        'RShoulderRoll': [0.0, 2.0, 0.0],
        'RElbowYaw': [80, -20, 0.0],
        'RElbowRoll': [0.0, 0.0, 0.0],
        'LHipYawPitch': [0.0, 50.0, -85.0],
        'LHipRoll': [0.0, 10, 0.0],
        'LHipPitch': [15, 0.0, 0.0],
	    'LKneePitch': [0.0, 0.0, -60],
        'LAnklePitch': [0.0, 0.0, -30],
        'LAnkleRoll': [0.0, 30, 0.0],
        'RHipYawPitch': [0.0, -30, 0.0],
        'RHipRoll': [1.0, 0.0, 0.0],
        'RHipPitch': [0.0, -10, 0.0],
        'RKneePitch': [0.0, 0.0, -50],
        'RAnklePitch': [0.0, 0.0, -70],
        'RAnkleRoll': [0.0, 0.0, 0.0]}

        joint_axis = {
        'HeadYaw': 'Z',
        'HeadPitch': 'Y',
        'RShoulderPitch': 'Y',
        'RShoulderRoll': 'Z',
        'RElbowYaw': 'X',
        'RElbowRoll': 'Z',
        'LShoulderPitch': 'Y',
        'LShoulderRoll': 'Z',
        'LElbowYaw': 'X',
        'LElbowRoll': 'Z',
        'LHipYawPitch': 'Y-Z',
        'RHipYawPitch': 'Y-Z',
        'LHipRoll': 'X',
        'LHipPitch': 'Y',
        'LKneePitch': 'Y',
        'LAnklePitch': 'Y',
        'LAnkleRoll': 'X',
        'RHipRoll': 'X',
        'RHipPitch': 'Y',
        'RKneePitch': 'Y',
        'RAnklePitch': 'Y',
        'RAnkleRoll': 'X',
        }

        if joint_axis[joint_name] = 'X':
            T[0:3, 0:3] = np.array([[1.0, 0.0, 0.0],
                                    [0.0, c, -s],
                                    [0.0, s, c]])
        elif joint_axis[joint_name] = 'Y':
            T[0:3, 0:3] = np.array([[c, 0.0, s],
                                    [0.0, 1.0, 0.0],
                                    [-s, 0.0, c]])
        elif joint_axis[joint_name] = 'Z':
            T[0:3, 0:3] = np.array([[c, -s, 0.0],
                                    [s, c, 0.0],
                                    [0.0, 0.0, 0.0]])
        elif joint_axis[joint_name] = 'Y-Z':
            T[0:3, 0:3] = np.array([[c, -(s/sqrt(2)), (s/sqrt(2))],
                                    [(s/sqrt(2)), (1.0/2.0)*(1+c), (1.0/2.0)*(1-c)],
                                    [-(s/sqrt(2)), (1.0/2.0)*(1-c), (1.0/2.0)*(1+c)]])

        T[0, 3] = offsets[joint_name][0]
        T[1, 3] = offsets[joint_name][1]
        T[2, 3] = offsets[joint_name][2]

        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                # YOUR CODE HERE
                T *= Tl

                self.transforms[joint] = T

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
