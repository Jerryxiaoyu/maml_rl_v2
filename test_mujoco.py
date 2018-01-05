from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.mujoco.ant_env_rand_Linedirec import AntEnvRandLineDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf
import numpy as np
from math import pi,sin,cos


env = AntEnvRandLineDirec()


a = np.random.uniform(-3.1415/2, 3.1415/2, (50, ))
learner_env_goals = env.sample_goals(50)
print(learner_env_goals)
xposafter = env.get_body_com("torso")[0]
yposafter = env.get_body_com("torso")[1]
print('xposafter= ',xposafter)
print('yposafter= ',yposafter)
theta = pi / 4

comvel_xy = np.array([xposafter, yposafter])
print('comvel_xy = ',comvel_xy)




point = np.array([-1,4])
theta = -pi/4
u = np.array([cos(theta),sin(theta)])
proj_par = point.dot(np.transpose(u))
proj_ver = abs(u[0] * point[1] - u[1] * point[0])
forward_reward = 1*proj_par + 1*proj_ver

print('proj_par = ', proj_par)
print('proj_ver = ', proj_ver)
print('forward_reward = ', forward_reward)

print(a)

env.reset()
for i in range(100):
    env.render()
   # step()
    action = env.action_space.sample()
    env.forward_dynamics(action)

    comvel = env.get_body_comvel("torso")


#    forward_reward = env.goal_direction * comvel[0]


    lb, ub = env.action_bounds
    scaling = (ub - lb) * 0.5
    ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
    contact_cost = 0.5 * 1e-3 * np.sum(
        np.square(np.clip(env.model.data.cfrc_ext, -1, 1))),
    survive_reward = 0.05
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    state = env._state
    notdone = np.isfinite(state).all() \
              and state[2] >= 0.2 and state[2] <= 1.0
    done = not notdone
    ob = env.get_current_obs()



    if not (i % 10):

        qpos = env.model.data.qpos
        qvel = env.model.data.qvel
        a = np.concatenate([qpos.flat[2:], qvel.flat])
        print('itr = ',i)
        #print('np.concatenate shape: %d \n' % (len(a)), a)
        print('comvel',comvel)

