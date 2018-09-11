
from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from rllab.envs.mujoco.half_cheetah_env_rand_direc import HalfCheetahEnvRandDirec
from rllab.envs.mujoco.half_cheetah_env_rand_disable import HalfCheetahEnvRandDisable
from rllab.envs.mujoco.half_cheetah_VaryingEnv import HalfCheetahVaryingEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv


import tensorflow as tf

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):

    @variant
    def fast_lr(self):                  # learning rate of policy  alpha
        return [0.1]

    @variant
    def meta_step_size(self):           # learning rate of meta-update  beta
        return [0.01]

    @variant
    def fast_batch_size(self):          # Number of samples per iteration
        return [20]  # #10, 20, 40

    @variant
    def meta_batch_size(self):          #Number of tasks sampled per meta-update
        return [40]

    @variant
    def seed(self):
        return [1]

    @variant
    def direc(self):                    # directionenv vs. terrain!!!! if False
        return [True]
    
   


# should also code up alternative KL thing

variants = VG().variants()

max_path_length = 200                    # Maximum length of a single rollout.
num_grad_updates = 1
n_itr = 800

use_maml=True

for v in variants:
    direc = v['direc']
    learning_rate = v['meta_step_size']

    if direc:
        env = TfEnv(normalize(HalfCheetahEnvRandDirec()))
    else:
        env = TfEnv(normalize(HalfCheetahEnvRandDisable()))
    policy = MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=v['fast_lr'],        # learning rate of policy
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = MAMLTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v['fast_batch_size'],        # Number of samples per iteration.
        max_path_length=max_path_length,        # Maximum length of a single rollout.
        meta_batch_size=v['meta_batch_size'],   # Number of tasks sampled per meta-update
        num_grad_updates=num_grad_updates,      # Number of fast gradient updates
        n_itr=n_itr,                            # Number of iterations.
        use_maml=use_maml,
        step_size=v['meta_step_size'],          # learning rate of meta-update
        plot=False,
    )
    direc = 'direc22' if direc else 'slope'

    run_experiment_lite(
        algo.train(),
        exp_prefix='trpo_maml_cheetah' + direc + str(max_path_length),
        exp_name='maml'+str(int(use_maml))+'_fbs'+str(v['fast_batch_size'])+'_mbs'+str(v['meta_batch_size'])+'_flr_' + str(v['fast_lr'])  + '_mlr' + str(v['meta_step_size']),
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="gap",
        snapshot_gap=25,
        sync_s3_pkl=True,
        python_command='python3',
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=v["seed"],
        mode="local",
        #mode="ec2",
        variant=v,
        # plot=True,
        # terminate_machine=False,
    )
