from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np
from math import pi,sin,cos

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class AntEnvRandLineDirec(MujocoEnv, Serializable):

    FILE = 'ant.xml'

    def __init__(self, goal=None, *args, **kwargs):
        self._goal_vel = goal
        super(AntEnvRandLineDirec, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def sample_goals(self, num_goals):
        # for fwd/bwd env, goal direc is backwards if < 1.5, forwards if > 1.5
        return np.random.uniform(-pi/2, pi/2, (num_goals, ))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_vel = reset_args
        if goal_vel is not None:
            self.goal_theta = goal_vel
        elif self._goal_vel is None:
            self.goal_theta = np.random.uniform(-pi/2, pi/2)


        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        comvel_xy = np.array([comvel[0], comvel[1]])
        u = np.array([cos(self.goal_theta), sin(self.goal_theta)])
        proj_par = comvel_xy.dot(np.transpose(u))
        proj_ver = abs(u[0] * comvel_xy[1] - u[1] * comvel_xy[0])
        forward_reward = 1 * proj_par - 0.5 * proj_ver




        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
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

"""""
comvel = self.get_body_comvel("torso")
comvel_xy = np.array([xposafter, yposafter])
u = np.array([cos(self.goal_theta), sin(self.goal_theta)])
proj_par = comvel_xy.dot(np.transpose(u))
proj_ver = abs(u[0] * comvel_xy[1] - u[1] * comvel_xy[0])
forward_reward = 1 * proj_par - 0.5 * proj_ver


##以下为读取位置求速度的方案
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        theta = self.goal_theta
        comvel_xy_before = np.array([xposbefore, yposbefore])
        u = np.array([cos(theta), sin(theta)])
        proj_parbefore = comvel_xy_before.dot(np.transpose(u))

        self.forward_dynamics(action)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]



        comvel_xy_after = np.array([xposafter, yposafter])
        # u = np.array([cos(theta), sin(theta)])
        proj_parafter = comvel_xy_after.dot(np.transpose(u))
        proj_ver = abs(u[0] * comvel_xy_after[1] - u[1] * comvel_xy_after[0])
        forward_reward = 1 * (proj_parafter - proj_parbefore) - 0.5 * proj_ver
"""""