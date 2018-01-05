from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv


import numpy as np
import tensorflow as tf

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def fast_lr(self):
        return [0.1]

    @variant
    def meta_step_size(self):
        return [0.01] # sometimes 0.02 better

    @variant
    def fast_batch_size(self):
        return [20,30]

    @variant
    def meta_batch_size(self):
        return [40,50] # at least a total batch size of 400. (meta batch size*fast batch size)

    @variant
    def seed(self):
        return [1]

    @variant
    def task_var(self):  # fwd/bwd task or goal vel task
        # 0 for fwd/bwd, 1 for goal vel (kind of), 2 for goal pose
        return [0]


# should also code up alternative KL thing

variants = VG().variants()
print(variants)


for v in variants:
    task_var = v['task_var']
    #oracle = v['oracle']


    print('v  :  ',v)
    print('task:  ',task_var)
    #print('oracle: ',oracle)
    print('---------------------')

make_video = True
'''''
if not make_video:
    test_num_goals = 10
    np.random.seed(2)
    goals = np.random.uniform(0.0, 3.0, size=(test_num_goals, ))
else:
    np.random.seed(1)
    test_num_goals = 2  
    goals = [0.0, 3.0]
    file_ext = 'mp4'  # can be mp4 or gif
print(goals)
'''''
np.random.seed(0)
np.random.rand(5)