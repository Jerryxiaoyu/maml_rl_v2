from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.cellrobot_rand_direc_env import CellRobotRandDirectEnv
from rllab.envs.mujoco.cellrobot_rand_direc2_env import CellRobotRandDirect2Env
from rllab.envs.mujoco.cellrobot_rand_direc_pi4_env import CellRobotRandDirectpi4Env
from rllab.envs.mujoco.cellrobot_rand_direc_pi4_env2 import CellRobotRandDirectpi4Env2
from rllab.envs.mujoco.cellrobot_rand_direc_env_body import CellRobotRandDirectBodyEnv
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
        return [0.02,0.05, ]  # sometimes 0.02 better
    
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
        return [3]


ssh_FLAG = False

exp_id = 9
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
        env = TfEnv(normalize(CellRobotRandDirectpi4Env()))
        task_var = 'directpi-4'
    elif task_var == 1:
        env = TfEnv(normalize(CellRobotRandDirectEnv()))
        task_var = 'direc'
    elif task_var == 2:
        env = TfEnv(normalize(CellRobotRandDirect2Env()))
        task_var = 'direc2'
    elif task_var == 3:
        env = TfEnv(normalize(CellRobotRandDirectpi4Env2()))  # -pi/4 固定 body
        task_var = 'direcpi-4-2'
    elif task_var == 4:
        env = TfEnv(normalize(CellRobotRandDirectBodyEnv()))    #利用body位置做sate
        task_var = 'direc-body'

    exp_name = 'Cellrobot_trpo_maml' + task_var + '_' + str(max_path_length) + '_EXP' + str(exp_id)

    filenames = glob.glob('*.py')  # put copy of all python files in log_dir
    for filename in filenames:  # for reference
        shutil.copy(filename, os.path.join('data/local', exp_name))
    
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
    
    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_name,
        exp_name='maml' + str(int(use_maml)) + '_fbs' + str(v['fast_batch_size']) + '_mbs' + str(
            v['meta_batch_size']) + '_flr_' + str(v['fast_lr']) + '_mlr' + str(v['meta_step_size']),
        # Number of parallel workers for sampling
        n_parallel=35,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=2,
        sync_s3_pkl=True,
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        # mode="ec2",
        use_gpu=False,
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
    
    if ssh_FLAG:
        local_dir = os.path.abspath('data/local/' + exp_name + '/')
        remote_dir = '/home/drl/PycharmProjects/DeployedProjects/CR_CPG/Hyper_lab/log-files/AWS_logfiles/' + exp_name + '/'
        ssh.upload(local_dir, remote_dir, hostname=hostname, port=port, username=username,
                   pkey_path=key_path)