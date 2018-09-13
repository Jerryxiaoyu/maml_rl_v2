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

state_M =np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                   [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])

position_vector = [0.9005710154022419, 0.19157649858525766, 0.20363844865472536, -0.2618038524762938, -0.04764016477204058, -0.4923544636213292, -0.30514082693887024, 0.7692727139092137, 0.7172509186944478, -0.6176943450166859, -0.43476218435592706, 0.7667223977603919, 0.29081693103406536, 0.09086369237435465, 0.0, 0.0, -0.0171052262902362, 0.0, 0.0, 0.0, 0.0, 0.0004205454597565903, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6989070655586036, 1.231416257452789, 1.188419262405775, -1.0974581723778125, -1.023151598620554, -0.40304458466288917, 0.5513169936393982, 0.646385738643396, 1.3694066886743392, 0.7519699447089043, 0.06997050535309216, -1.5500743998481212, 0.8190474090403703]

class CellRobotRandDirectEnv(MujocoEnv, Serializable):
    FILE = 'cellrobot_Quadruped_float.xml'
    def __init__(self, goal_num=None, *args, **kwargs):
       
        self.goal_num = goal_num

        self.goal_theta = 0.0
        self.quat_init = [0.49499825, -0.49997497, 0.50500175, 0.49997499]
        self.t = 0
        self.CPG_controller = CPG_network(position_vector)
        
        super(CellRobotRandDirectEnv, self).__init__(*args, **kwargs)
       
        Serializable.__init__(self, *args, **kwargs)
     
        self.reset(reset_args=goal_num)
        
    def sample_goals(self, num_goals):
        # for fwd/bwd env, goal direc is backwards if < 1.5, forwards if > 1.5
        return np.random.uniform(-pi/3, pi/3, (num_goals, ))
        
        
    #
    # def step(self, a):
    #
    #     action = self.CPG_transfer(a, self.CPG_controller )
    #
    #     xposbefore = self.get_body_com("torso")[0]
    #     yposbefore = self.get_body_com("torso")[1]
    #     theta = self.goal_theta
    #     comvel_xy_before = np.array([xposbefore, yposbefore])
    #     u = np.array([cos(theta), sin(theta)])
    #     proj_parbefore = comvel_xy_before.dot(np.transpose(u))
    #
    #     self.do_simulation(action, self.frame_skip)
    #
    #     xposafter = self.get_body_com("torso")[0]
    #     yposafter = self.get_body_com("torso")[1]
    #
    #     comvel_xy_after = np.array([xposafter, yposafter])
    #     # u = np.array([cos(theta), sin(theta)])
    #     proj_parafter = comvel_xy_after.dot(np.transpose(u))
    #     proj_ver = abs(u[0] * comvel_xy_after[1] - u[1] * comvel_xy_after[0])
    #     forward_reward = 1 * (proj_parafter - proj_parbefore)/ self.dt  - 5 * proj_ver  #/ self.dt
    #     #forward_reward = 1 *  proj_parafter   - 5 * proj_ver  # / self.dt
    #
    #     ctrl_cost = 0
    #     contact_cost = 0
    #     survive_reward = 0
    #     reward = forward_reward+survive_reward
    #     state = self.state_vector()
    #     notdone = np.isfinite(state).all() \
    #               and state[2] >= 0.1 and state[2] <= 0.6
    #     done = not notdone
    #     # done = False
    #     ob = self._get_obs()
    #     self.t += 1
    #     return ob, reward, done, dict(
    #         reward_forward=forward_reward,
    #         reward_ctrl=-ctrl_cost,
    #         reward_contact=-contact_cost,
    #         reward_survive=survive_reward)
    #
   
    
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
        #print(self.goal_theta)
        self.goal_direction = -1.0 if self._goal_vel < 1.5 else 1.0
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs

    def step(self, a):
        #print(a)
        u = np.array([cos(self.goal_theta), sin(self.goal_theta)])
        action = self.CPG_transfer(a, self.CPG_controller)
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        
        comvel_xy_before = np.array([xposbefore, yposbefore])
        proj_parbefore = comvel_xy_before.dot(np.transpose(u))
        
        self.forward_dynamics(action)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]

        comvel_xy_after = np.array([xposafter, yposafter])
        proj_parafter = comvel_xy_after.dot(np.transpose(u))
        
        comvel = self.get_body_comvel("torso")
        comvel_xy = np.array([comvel[0], comvel[1]])
        
        proj_par = comvel_xy.dot(np.transpose(u))
        proj_ver = abs(u[0] * comvel_xy[1] - u[1] * comvel_xy[0])
        #forward_reward = 1* proj_par - 10 * proj_ver

        #print('reward: ', (proj_parafter - proj_parbefore) /0.01, 5 * proj_ver)
        forward_reward = 1 * (proj_parafter - proj_parbefore) /0.01 - 10 * proj_ver
        
        # lb, ub = self.action_space_ture.bounds
        # scaling = (ub - lb) * 0.5
        # ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        ctrl_cost=0
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05
        #print('reward: ', forward_reward,-ctrl_cost, -contact_cost )
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

    def CPG_transfer(self,RL_output, CPG_controller ):
        #print(RL_output)
        CPG_controller.update(RL_output)
        # if self.t % 100 == 0:
        #     #CPG_controller.update(RL_output)
        #     print(RL_output)
        ###adjust CPG_neutron parm using RL_output
        output_list = CPG_controller.output(state=None)
        target_joint_angles = np.array(output_list[1:])# CPG 第一个输出为placemarke
        cur_angles = np.concatenate([state_M.dot(self.model.data.qpos[7:].reshape((-1, 1))).flat])
        action = PID_controller(cur_angles, target_joint_angles)
        return action