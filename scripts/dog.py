import pybullet as p
import time
import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf as kdl_parser
import robotics_math as rm

class Dog:
    def prepare_kin_dyn(self):
        flag, self.tree = kdl_parser.treeFromFile("data/laikago/laikago.urdf")

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
        # if self.debug:
            # self.debug_count += 1
            # if (self.debug_count % 10) == 0:
                # frame = kdl.Frame()
                # self.fk_ee.JntToCart(self.joints, frame)
#
                # print " "
                # print " "
                # print "frame translation:"
                # print frame.p
                # print "frame rotation:"
                # print frame.M
                # print " "
#
                # jacobian = kdl.Jacobian(self.num_joints)
                # self.jac_ee.JntToJac(self.joints, jacobian)
                # print "jacobian"
                # print jacobian

    def __init__(self, quadruped):
        self.q = quadruped
        # self.base_link = p.getLinkState(self.q, 0)
        self.base_link = "chasis"
        self.bl_link   = "BL_lower_leg"
        self.br_link   = "BR_lower_leg"
        self.fl_link   = "FL_lower_leg"
        self.fr_link   = "FR_lower_leg"
        self.prepare_kin_dyn()

        self.kpp = 10
        self.kdp = 1

        self.kpw = 10
        self.kdw = 1


        #enable collision between lower legs

        for j in range (p.getNumJoints(quadruped)):
                print(p.getJointInfo(quadruped,j))

        #2,5,8 and 11 are the lower legs
        lower_legs = [2,5,8,11]
        for l0 in lower_legs:
            for l1 in lower_legs:
                if (l1>l0):
                    enableCollision = 1
                    print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
                    p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

        jointIds=[]
        paramIds=[]
        jointOffsets=[]
        jointDirections=[-1,1,1,1,1,1,-1,1,1,1,1,1]
        jointAngles=[0,0,0,0,0,0,0,0,0,0,0,0]

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

    def compute_balance(self, pos_des, vel_des, ori_des, ang_des, f_prev, stance):
        p_c = np.array(0 ,0, 0)
        p_1 = np.array(0 ,0, 0)
        p_2 = np.array(0 ,0, 0)
        p_3 = np.array(0 ,0, 0)
        p_4 = np.array(0 ,0, 0)

        vel = 0
        curr_ori = np.ones((3, 3))
        curr_angular = np.zeros((3, 1))

        acc         = self.kpp * (pos_des - p_c) + self.kdp * (vel_des - vel)
        angular_acc = self.kpw * np.log(ori_des.dot(curr_ori)) + self.kdw * (ang_des - curr_angular)

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
        alpha = 0.1
        beta = 0.1

        Atop = np.hstack((eye3, eye3, eye3, eye3))
        Abot = np.hstack((skew_3d(p1-p_c), skew_3d(p2-p_c), skew_3d(p3-p_c), skew_3d(p4-p_c)))
        A = np.hstack(Atop, Abot)

        S = np.eye(6)
        g = np.array([0, 0, -9.8])

        bd_top = m  * (acc + g)
        bd_bot = Ig * (angular_acc)
        bd = np.vstack((bd_top, bd_bot))

        P =  2 * (np.dot(np.dot( A.T, S), A) + (alpha + beta) * np.eye(12))
        q = -2 * (np.dot(np.dot(bd.T, S), A) + beta * f_prev)

        G = np.array(constraints)
        num_h = G.shape[0]
        h = np.zeros((num_h, 1))
        F_opt = quadprog_solve_qp(P, q, G=G, h=h)

        return F_opt

    def get_state(self):
        return None

    def force_to_torques(self, j_torques):
        # TODO do some jacobian stuff
        torques = j_torques
        # to jacobian stuff

        return torques

    def compute_action(self, pos_des, vel_des, ori_des, ang_des):
        state =  self.get_state
        f_prev = 0

        foot_forces = self.compute_balance(pos_des, vel_des, ori_des, ang_des, f_prev, stance)
        joint_torques = 0
        joint_torques = self.force_to_torques(joint_torques)
        action = joint_torques
        return action

    def apply_action(self, action):
        for j in range (12):
            targetPos = float(action[j])
            p.setJointMotorControl2(quadruped, self.jointIds[j], p.POSITION_CONTROL, jointDirections[j]*targetPos+jointOffsets[j], force=maxForce)

    def apply_action_pos(self, action):
        for j in range (12):
            targetPos = float(action[j])
            p.setJointMotorControl2(quadruped, self.jointIds[j], p.POSITION_CONTROL, jointDirections[j]*targetPos+jointOffsets[j], force=maxForce)


if __name__=="__main__":
    rate = 500
    p.connect(p.GUI)
    plane = p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./rate)
    #p.setDefaultContactERP(0)
    #urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    urdfFlags = p.URDF_USE_SELF_COLLISION
    StartOrientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
    quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,.5], StartOrientation, flags = urdfFlags,useFixedBase=False)
    # quadruped = p.loadURDF("laikago/laikago.urdf",[0,0,.5],[0,0.5,0.5,0], flags = urdfFlags,useFixedBase=False)
    d = Dog(quadruped)

    position_desired    = np.array([0, 0, 0.5]) # x, y, z
    velocity_desired    = np.array([0, 0, 0.0]) # x, y, z
    orientation_desired = rm.rot_rpy(np.pi/2, 0, 0.0) # r, p, y
    angular_desired     = np.array([0, 0, 0])   # omega

    while(1):
        action = d.compute_action(position_desired, velocity_desired, orientation_desired, angular_desired)
        d.apply_action(action)
        p.stepSimulation()
        time.sleep(1./rate)
