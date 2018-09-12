import numpy as np
from gym import utils

from math import pi,sin,cos
import numpy as np

from rllab.misc import autoargs
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
#from .mujoco_env import MujocoEnv

from CPG_core.PID_controller import PID_controller

from CPG_core.math.transformation import euler_from_quaternion,quaternion_inverse ,quaternion_multiply

# choose your CPG network
# from CPG_core.controllers.CPG_controller_quadruped_sin import CPG_network
from CPG_core.controllers.CPG_controller_quadruped_sin import CPG_network

state_M = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1.]])

position_vector = [0.6429382282419057, -1.504515744295227, -1.2122146768896522, -1.5430866210080398, -0.40339625754811115, 1.797866485292635, 1.8465815905382332, 1.3156099280263092, 1.1815248128471576, 0.4557141890654699, -1.3136558716839821, -1.383742714528093, 1.0666337767012442, -0.46609084892228037, 0.5266071808950659, 0.008279403737123438, 0.0, 0.0, 0.0, 0.011237911878610663, -0.0146826210838802, 0.0, 0.0, 0.0, 0.0, -0.017814994116624273, 0.0, 0.018121282385488727, 0.0, -0.3242713509141666, -0.5905465210728494, -1.5530731911249718, 0.7060008434892526, 0.6718690361326529, 0.30814153016454116, -0.2900699626568739, -1.4214811438222459, -0.8181964756164031, -0.9037143779342285, -0.6716727364566586, -1.0711308729593805, -1.464835073477411, 0.2443659340371438]

CPG_node_num = 14
from CPG_core.controllers.CPG_controller_bigdog2_sin import CPG_network

class CellRobotBigDog2RandDirectEnv(MujocoEnv, Serializable):
    FILE = 'cellrobot_BigDog2_float.xml'
    def __init__(self, goal_num=None, *args, **kwargs):
       
        self.goal_num = goal_num
        self.goal_theta = 0.0
        self.quat_init = [0.49499825, -0.49997497, 0.50500175, 0.49997499]
        self.t = 0
        self.CPG_controller = CPG_network(CPG_node_num, position_vector)
        
        super(CellRobotBigDog2RandDirectEnv, self).__init__(*args, **kwargs)
       
        Serializable.__init__(self, *args, **kwargs)
     
        self.reset(reset_args=goal_num)
        
    def sample_goals(self, num_goals):
        # for fwd/bwd env, goal direc is backwards if < 1.5, forwards if > 1.5
        return np.random.uniform(-pi/3, pi/3, (num_goals, ))
 
    def get_current_obs(self):
        quat = self.model.data.qpos.flat[3:7]
        # print('quat = ', quat)
        quat_tranfor = quaternion_multiply(quat, quaternion_inverse(self.quat_init))
        angle = euler_from_quaternion(quat_tranfor, 'rxyz')
        
        #print(self.goal_theta)
        return np.concatenate([
            self.get_body_com("torso").flat,
            # self.sim.data.qpos.flat[:3],  # 3:7 表示角度
            # self.sim.data.qpos.flat[:7],  # 3:7 表示角度
            np.array(angle),
            np.array([angle[2] - self.goal_theta])
        ]).reshape(-1)
    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_vel = reset_args
        if goal_vel is not None:
            self._goal_vel = goal_vel
        else:
            self._goal_vel = np.random.uniform(-pi/3, pi/3)

        self.goal_theta = self._goal_vel
        self.goal_direction = -1.0 if self._goal_vel < 1.5 else 1.0
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def step(self, a):
        action = self.CPG_transfer(a )
        
        
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        comvel_xy = np.array([comvel[0], comvel[1]])
        u = np.array([cos(self.goal_theta), sin(self.goal_theta)])
        proj_par = comvel_xy.dot(np.transpose(u))
        proj_ver = abs(u[0] * comvel_xy[1] - u[1] * comvel_xy[0])
        forward_reward = 5* proj_par - 1 * proj_ver

        # lb, ub = self.action_space_ture.bounds
        # scaling = (ub - lb) * 0.5
        # ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        ctrl_cost=0
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        
        
        state = self._state
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.1 and state[2] <= 0.6
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

    def CPG_transfer(self,RL_output  ):
        #print(RL_output)
        self.CPG_controller.update(RL_output)
        # if self.t % 100 == 0:
        #     #CPG_controller.update(RL_output)
        #     print(RL_output)
        ###adjust CPG_neutron parm using RL_output
        output_list = self.CPG_controller.output(state=None)

        target_joint_angles = np.array([output_list[1], 0, 0, 0, output_list[2],
                                        output_list[3], output_list[4], output_list[5], output_list[6],
                                        output_list[7], output_list[8], output_list[9], output_list[10],
                                        output_list[11], output_list[12], output_list[13], output_list[14], ])
        
         
        cur_angles = np.concatenate([state_M.dot(self.model.data.qpos[7:].reshape((-1, 1))).flat])
        action = PID_controller(cur_angles, target_joint_angles)
        return action