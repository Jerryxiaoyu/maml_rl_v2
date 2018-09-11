from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.cellrobot_rand_direc_env import CellRobotRandDirectEnv
from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy

#from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_2 import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

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
        return [20]

    @variant
    def meta_batch_size(self):
        return [40] # at least a total batch size of 400. (meta batch size*fast batch size)

    @variant
    def seed(self):
        return [1]

    @variant
    def task_var(self):  # fwd/bwd task or goal vel task
        # 0 for fwd/bwd, 1 for goal vel (kind of), 2 for goal pose
        return [1]


# should also code up alternative KL thing

variants = VG().variants()

max_path_length = 200
num_grad_updates = 1
use_maml=True
exp_id = 1

for v in variants:
    task_var = v['task_var']

    if task_var == 0:
        env = TfEnv(normalize(AntEnvRandDirec()))
        task_var = 'lalalala'
    elif task_var == 1:
        env = TfEnv(normalize(CellRobotRandDirectEnv()))
        task_var = 'direc'
    elif task_var == 2:
        env = TfEnv(normalize(AntEnvRandGoal()))
        task_var = 'papapap'
    policy = MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=v['fast_lr'],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.sigmoid,
        hidden_sizes=(64,64),
    
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = MAMLTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['fast_batch_size'], # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=num_grad_updates,
        n_itr=800,
        use_maml=use_maml,
        step_size=v['meta_step_size'],
        plot=False,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix='Cellrobot_trpo_maml' + task_var + '_' + str(max_path_length) ,
        exp_name='maml'+str(int(use_maml))+'_fbs'+str(v['fast_batch_size'])+'_mbs'+str(v['meta_batch_size'])+'_flr_' + str(v['fast_lr'])  + '_mlr' + str(v['meta_step_size'])+'_EXP'+str(exp_id),
        # Number of parallel workers for sampling
        n_parallel=71,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=10,
        sync_s3_pkl=True,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        #mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
