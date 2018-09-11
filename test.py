import rllab.mujoco_py
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.point_env import PointEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.gather.point_gather_env import PointGatherEnv
from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
from rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv



from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_goal_oracle import AntEnvRandGoalOracle

env = AntGatherEnv()
ob_space = env.observation_space
act_space = env.action_space
ob = env.reset()
assert ob_space.contains(ob)



for _ in range(5000):
    a = act_space.sample()
    res = env.step(a)
    env.render()