import pybullet as p
import time
import numpy as np
from scipy.linalg import block_diag
import PyKDL as kdl
import kdl_parser_py.urdf as kdl_parser
import robotics_math as rm
import gait as gaits
from leg_kin import Leg

MOTOR_NAMES = [
        "FL_hip_joint", "FL_upper_leg_joint", "FL_lower_leg_joint", # leg 1
        "FR_hip_joint", "FR_upper_leg_joint", "FR_lower_leg_joint", # leg 2
        "RL_hip_joint", "RL_upper_leg_joint", "RL_lower_leg_joint", # leg 3
        "RR_hip_joint", "RR_upper_leg_joint", "RR_lower_leg_joint"  # leg 4
]

class Dog:
    def prepare_kin_dyn(self):
        flag, self.tree = kdl_parser.treeFromFile("data/jelly/jelly.urdf")

        chain = self.tree.getChain(self.base_link, self.bl_link)
        self.fk_bl = kdl.ChainFkSolverPos_recursive(chain)
        self.jac_bl = kdl.ChainJntToJacSolver(chain)

        chain = self.tree.getChain(self.base_link, self.br_link)
        self.fk_br = kdl.ChainFkSolverPos_recursive(chain)
        self.jac_br = kdl.ChainJntToJacSolver(chain)

        chain = self.tree.getChain(self.base_link, self.fl_link)
        self.fk_fl = kdl.ChainFkSolverPos_recursive(chain)
        self.jac_fl = kdl.ChainJntToJacSolver(chain)

        chain = self.tree.getChain(self.base_link, self.fr_link)
        self.fk_fr = kdl.ChainFkSolverPos_recursive(chain)
        self.jac_fr = kdl.ChainJntToJacSolver(chain)

        # convention front_right, front_left, back_right, back_left
        self.jac_list = [self.jac_fl, self.jac_fr, self.jac_bl, self.jac_br]
        self.fk_list  = [self.fk_fl , self.fk_fr , self.fk_bl , self.fk_br ]

        joints = kdl.JntArray(3)
        joints[0] = 0
        joints[1] = 0
        joints[2] = 0
        frame = kdl.Frame()
        jk_list = self.fk_list
        print(jk_list[0].JntToCart(joints, frame))
        print(frame)
        print(jk_list[1].JntToCart(joints, frame))
        print(frame)
        print(jk_list[2].JntToCart(joints, frame))
        print(frame)
        print(jk_list[3].JntToCart(joints, frame))
        print(frame)

    def __init__(self, quadruped):
        self.q = quadruped
        # self.base_link = p.getLinkState(self.q, 0)
        self.base_link = "chassis"
        self.bl_link   = "RL_lower_leg"
        self.br_link   = "RR_lower_leg"
        self.fl_link   = "FL_lower_leg"
        self.fr_link   = "FR_lower_leg"
        self.prepare_kin_dyn()
        self._BuildJointNameToIdDict()
        self._BuildMotorIdList()

        self.kpp = 1000
        self.kdp = 100

        self.kpw = 200
        self.kdw = 20


        #enable collision between lower legs
        for j in range (p.getNumJoints(quadruped)):
                print(p.getJointInfo(quadruped,j))

        #2,5,8 and 11 are the lower legs
        lower_legs = [2,5,8,11]
        for l0 in lower_legs:
            for l1 in lower_legs:
                if (l1>l0):
                    enableCollision = 1
                    print("collision for pair",l0,l1,p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
                    p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

        jointIds=[]
        paramIds=[]
        jointOffsets=[]
        jointDirections=[-1,1,1,1,1,1,-1,1,1,1,1,1]
        jointAngles=[0,0,0,0,0,0,0,0,0,0,0,0]
        self._motor_direction = jointDirections
        self.jointDirections = jointDirections

        for i in range (4):
            jointOffsets.append(0)
            jointOffsets.append(-0.7)
            jointOffsets.append(0.7)

        maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

        for j in range (p.getNumJoints(quadruped)):
            p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
            info = p.getJointInfo(quadruped,j)
            print(info)
            jointName = info[1]
            jointType = info[2]
            if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
                jointIds.append(j)
        self.jointIds = jointIds

        p.getCameraImage(480,320)
        p.setRealTimeSimulation(0)

        # joints=[]
        # while(1):
            # with open("data1.txt","r") as filestream:
                # for line in filestream:
                    # maxForce = p.readUserDebugParameter(maxForceId)
                    # currentline = line.split(",")
                    # frame = currentline[0]
                    # t = currentline[1]
                    # joints=currentline[2:14]
                    # for j in range (12):
                        # targetPos = float(joints[j])
                        # p.setJointMotorControl2(quadruped,jointIds[j],p.POSITION_CONTROL,jointDirections[j]*targetPos+jointOffsets[j], force=maxForce)
#
                    # p.stepSimulation()
                    # for lower_leg in lower_legs:
                        # print("points for ", quadruped, " link: ", lower_leg)
                        # pts = p.getContactPoints(quadruped,-1, lower_leg)
                        # print("num points=",len(pts))
                        # for pt in pts:
                           # print(pt[9])
                           # print(pt)
#
                    # time.sleep(1./500.)
    def _BuildJointNameToIdDict(self):
        num_joints = p.getNumJoints(self.q)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.q, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildMotorIdList(self):
        self._motor_id_list = [
            self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES
        ]

    def GetBaseOrientation(self):
        """Get the orientation of minitaur's base, represented as quaternion.
        Returns:
          The orientation of minitaur's base.
        """
        _, orientation = (
            p.getBasePositionAndOrientation(self.q))
        return orientation

    def GetBasePosition(self):
        """Get the position of dog's base.
        Returns:
          The position of dog's base.
        """
        position, _ = (
            p.getBasePositionAndOrientation(self.q))
        return position

    def GetMotorAngles(self):
        """Get the eight motor angles at the current moment.
        Returns:
          Motor angles.
        """
        motor_angles = [
            p.getJointState(self.q, motor_id)[0]
            for motor_id in self._motor_id_list
        ]
        motor_angles = np.multiply(motor_angles, self._motor_direction)
        return motor_angles

    def GetMotorVelocities(self):
        """Get the velocity of all eight motors.
        Returns:
          Velocities of all eight motors.
        """
        motor_velocities = [
            p.getJointState(self.q, motor_id)[1]
            for motor_id in self._motor_id_list
        ]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def GetState(self):
        """Get the observations of minitaur.
        It includes the angles, velocities, torques and the orientation of the base.
        Returns:
          The observation list. observation[0:8] are motor angles. observation[8:16]
          are motor velocities, observation[16:24] are motor torques.
          observation[24:28] is the orientation of the base, in quaternion form.
        """
        pos = self.GetMotorAngles().tolist()
        vel = self.GetMotorVelocities().tolist()
        # self.GetMotorTorques().tolist()
        com_pos = list(self.GetBasePosition())
        com_ori = list(self.GetBaseOrientation())
        return com_pos, com_ori, pos, vel

    def compute_balance(self, pos_des, vel_des, ori_des, ang_des, f_prev, stance):

        p_c, curr_ori, joint_pos, joint_vel = self.GetState()
        curr_ori = rm.quat_to_rot(curr_ori)

        p_ = []
        for i in range(4):
            frame = kdl.Frame()

            joints = kdl.JntArray(3)
            joints[0] = joint_pos[i+ 0]
            joints[1] = joint_pos[i+ 1]
            joints[2] = joint_pos[i+ 2]
            self.fk_list[i].JntToCart(joints, frame)

            frame_p = np.array([frame.p[0], frame.p[1], frame.p[2]])

            foot_pos = p_c + frame_p
            p_.append(foot_pos)


        vel = 0
        curr_ori = np.ones((3, 3))
        curr_angular = np.zeros((3, 1))


        acc         = self.kpp * (pos_des - p_c) + self.kdp * (vel_des - vel)
        print("acceaccell: {}".format(acc))

        orientation_error = 0
        # orientation_error = np.log(ori_des.dot(curr_ori))

        angular_acc = self.kpw * orientation_error + self.kdw * (ang_des.reshape(3,1) - curr_angular)
        print("angular accel: {}".format(angular_acc))

        constraints = []
        theta = np.pi/4
        for s in stance:
            if s:
                cons = np.array([
                    [ 1, 0, -np.tan(theta)],
                    [-1, 0, -np.tan(theta)],
                    [0,  1, -np.tan(theta)],
                    [0, -1, -np.tan(theta)],
                    [0, 0, -1]
                ])
            else:
                print("error")
                cons = np.array([
                    [ 1, 0,  0],
                    [-1, 0,  0],
                    [0,  1,  0],
                    [0, -1,  0],
                    [0,  0,  1],
                    [0,  0, -1]
                ])
            constraints.append(cons)

        eye3 = np.eye(3)
        alpha = 0.001
        beta = 0.001

        Atop = np.hstack((eye3, eye3, eye3, eye3))
        Abot = np.hstack((rm.skew_3d(p_[0]-p_c), rm.skew_3d(p_[1]-p_c), rm.skew_3d(p_[2]-p_c), rm.skew_3d(p_[3]-p_c)))
        A = np.vstack([Atop, Abot])

        S = 10*np.eye(6)
        g = np.array([0, 0, -9.8])

        m = 25.0
        Ig = 1.1*np.eye(3)

        bd_top = m  * (acc + g)
        angular_acc = angular_acc.reshape(3,1)
        bd_bot = Ig.dot(angular_acc)
        bd = np.vstack((bd_top.reshape(3,1), bd_bot))

        P =  2 * (np.dot(np.dot( A.T, S), A) + (alpha + beta) * np.eye(12))
        q = -2 * (np.dot(np.dot(bd.T, S), A) + beta * f_prev)

        G = block_diag(*constraints)

        num_h = G.shape[0]
        h = np.zeros(num_h)
        q = q.reshape(-1)


        F_opt = rm.quadprog_solve_qp(P, q, G=G, h=h)
        # print(F_opt)

        # F_opt = np.linalg.solve(np.dot(A.T, A) + 0.1*np.eye(12),np.dot(A.T, bd))
        # print(F_opt)
        return F_opt

    def force_to_torques(self, forces, joint_positions):
        # TODO test (correct reference frame?????)
        torques = []
        i = 0
        for f, j in zip(forces, joint_positions):
            jacobian = kdl.Jacobian(3)
            joints = kdl.JntArray(3)
            joints[0] = joint_positions[i+ 0]
            joints[1] = joint_positions[i+ 1]
            joints[2] = joint_positions[i+ 2]
            self.jac_list[i].JntToJac(joints, jacobian)
            jacobian = rm.jac_to_np(jacobian)
            f.reshape(3,1)
            f = np.vstack([f.reshape(3,1), np.zeros((3,1))])
            t = (jacobian.T).dot(f)
            torques.append(t)
            i += 1
        return torques

    def compute_action(self, pos_des, vel_des, ori_des, ang_des, stance):
        # state =  self.get_state
        f_prev = 0
        foot_forces = self.compute_balance(pos_des, vel_des, ori_des, ang_des, f_prev, stance)
        foot_forces = foot_forces.reshape(4, 3)
        p_c, curr_ori, joint_pos, joint_vel = self.GetState()
        joint_torques = self.force_to_torques(foot_forces, joint_pos)
        return np.array(joint_torques).flatten()

    def apply_action(self, action):
        print(action)
        for j in range (12):
            targetForce = float(action[j])
            p.setJointMotorControl2(self.q, self.jointIds[j], p.TORQUE_CONTROL, force=targetForce)

    def apply_action_pos(self, action):
        for j in range (12):
            targetPos = float(action[j])
            p.setJointMotorControl2(self.q, self.jointIds[j], p.POSITION_CONTROL, targetPos, force=1000)


if __name__=="__main__":
    rate = 500
    p.connect(p.GUI)
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./rate)
    #p.setDefaultContactERP(0)
    #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    urdfFlags = p.URDF_USE_SELF_COLLISION
    StartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    quadruped = p.loadURDF("jelly/jelly.urdf",[0,0,.55], StartOrientation, flags = urdfFlags,useFixedBase=False)
    # quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)

    d = Dog(quadruped)

    position_desired    = np.array([0, 0, 0.55]) # x, y, z
    velocity_desired    = np.array([0, 0, 0.0]) # x, y, z
    orientation_desired = rm.rot_rpy(0, 0, 0.0) # r, p, y
    angular_desired     = np.array([0, 0, 0])   # omega

    p_c, curr_ori, joint_pos, joint_vel = d.GetState()
    print(joint_pos)

    zeros = np.array([0, 0, 0] * 4)
    home_joint_position = np.array([0, 1.2, -2.4] * 4)
    # time_r = 1000
    # for i in range(time_r):
        # d.apply_action_pos(zeros*(time_r-i)/float(time_r) + i/float(time_r) * home_joint_position)
        # p.stepSimulation()
        # time.sleep(1./rate)
#
    # time_r = 1000
    # for i in range(200):
        # d.apply_action_pos(home_joint_position)
        # p.stepSimulation()
        # time.sleep(1./rate)
#
    # stand_joint_position = np.array([0, 0.6, -1.2] * 4)
#
    # time_r = 1000
    # for i in range(time_r):
        # d.apply_action_pos(home_joint_position*(time_r-i)/float(time_r) + i/float(time_r) * stand_joint_position)
        # p.stepSimulation()
        # time.sleep(1./rate)

    ######################### Walking ###################################
    offset = 0.07
    p1 = np.array([offset + 1.85e-01, 4.5e-02, -3.94e-01])
    p2 = np.array([-1.85e-01, 4.5e-02, -3.94e-01])
    gait_controller = gaits.SimpleWalkingGait(0.85, p1, p2, mode="reverse_crab")
    T_cycle = 800
    ##########################################################################

    ######################### SideStep ###################################
    # offset = 0.0
    # p1 = np.array([offset ,-0.1 +  0.1e-01, -4.24e-01])
    # p2 = np.array([offset, -0.1 + -0.1e-01, -4.24e-01])
    # gait_controller = gaits.SimpleSideGait(0.85, p1, p2, mode="reverse_crab")
    # gait_controller.set_height(0.20)
    # T_cycle = 2400
    ############################################################################

    ########################## Bounding  ###################################
    offset = 0.07
    p1 = np.array([offset + 0.95e-01, -0.5e-01, -3.94e-01])
    p2 = np.array([-1.25e-01        , -0.5e-01, -3.94e-01])
    gait_controller = gaits.BoundGait(p1, p2, mode=None)
    gait_controller.set_height(0.09)
    T_cycle = 300
    ##############################################################################

    joints = []
    start_walk = gait_controller.step(0.0)
    print("Computed Joints")
    print(start_walk)

    time_r = 1000
    for i in range(time_r):
        d.apply_action_pos(zeros*(time_r-i)/float(time_r) + i/float(time_r) * start_walk)
        p.stepSimulation()
        time.sleep(1./rate)

    time_inc = 1.0/T_cycle
    normalized_time = 0

    reverse_time =  rate * 5
    while True:
        for _ in range(reverse_time):
            normalized_time = (normalized_time + time_inc)% 1.0
            joints = gait_controller.step(normalized_time)
            d.apply_action_pos(joints)
            p.stepSimulation()
            time.sleep(1./rate)
        for _ in range(reverse_time):
            normalized_time = (normalized_time - time_inc)% 1.0
            joints = gait_controller.step(normalized_time)
            d.apply_action_pos(joints)
            p.stepSimulation()
            time.sleep(1./rate)
    # while True:
        # stance = [1,1,1,1]
        # action = d.compute_action(position_desired,
                                  # velocity_desired,
                                  # orientation_desired,
                                  # angular_desired,
                                  # stance)
        # d.apply_action(action)
        # p.stepSimulation()
        # time.sleep(1./rate)
