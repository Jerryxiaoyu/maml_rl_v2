from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.cellrobot_Bigdog_rand_direc_env import CellRobotBigDog2RandDirectEnv
from rllab.envs.mujoco.ant_env_rand_goal import AntEnvRandGoal
from rllab.envs.mujoco.ant_env_rand_direc import AntEnvRandDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy

# from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_2 import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

import os
from datetime import datetime
import shutil
import glob

import vendor.ssh as ssh

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    
    @variant
    def fast_lr(self):
        return [0.1]
    
    @variant
    def meta_step_size(self):
        return [0.01, 0.02]  # sometimes 0.02 better
    
    @variant
    def fast_batch_size(self):
        return [20]
    
    @variant
    def meta_batch_size(self):
        return [20]  # at least a total batch size of 400. (meta batch size*fast batch size)
    
    @variant
    def seed(self):
        return [1]
    
    @variant
    def task_var(self):  # fwd/bwd task or goal vel task
        # 0 for fwd/bwd, 1 for goal vel (kind of), 2 for goal pose
        return [1]


ssh_FLAG = True

exp_id = 1
variants = VG().variants()
num = 0
for v in variants:
    num += 1
    print('exp{}: '.format(num), v)

# SSH Config
hostname = '2402:f000:6:3801:ee1c:d67d:4f92:55ad'  # '2600:1f16:e7a:a088:805d:16d6:f387:62e5'
username = 'drl'
key_path = '/home/ubuntu/.ssh/id_rsa_dl'

port = 22
# should also code up alternative KL thing

variants = VG().variants()

max_path_length = 500
num_grad_updates = 1
use_maml = True

for v in variants:
    task_var = v['task_var']
    
    if task_var == 0:
        env = TfEnv(normalize(AntEnvRandDirec()))
        task_var = 'lalalala'
    elif task_var == 1:
        env = TfEnv(normalize(CellRobotBigDog2RandDirectEnv()))
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
        hidden_sizes=(64, 64),
    
    )
    
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = MAMLTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['fast_batch_size'],  # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=v['meta_batch_size'],
        num_grad_updates=num_grad_updates,
        n_itr=800,
        use_maml=use_maml,
        step_size=v['meta_step_size'],
        plot=False,
    )
    exp_name = 'Cellrobot_BigDog2trpo_maml' + task_var + '_' + str(max_path_length) + '_EXP' + str(exp_id)
    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_name,
        exp_name='maml' + str(int(use_maml)) + '_fbs' + str(v['fast_batch_size']) + '_mbs' + str(
            v['meta_batch_size']) + '_flr_' + str(v['fast_lr']) + '_mlr' + str(v['meta_step_size']),
        # Number of parallel workers for sampling
        n_parallel=71,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=2,
        sync_s3_pkl=True,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        # mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
    
    if ssh_FLAG:
        local_dir = os.path.abspath('data/local/' + exp_name + '/')
        remote_dir = '/home/drl/PycharmProjects/maml_rl-master/data/AWS_data/' + exp_name + '/'
        ssh.upload(local_dir, remote_dir, hostname=hostname, port=port, username=username,
                   pkey_path=key_path)