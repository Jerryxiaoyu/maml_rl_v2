from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
from rllab.envs.mujoco.ant_env_oracle import AntEnvOracle
from rllab.envs.mujoco.ant_env_rand_Linedirec import AntEnvRandLineDirec
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

#from rllab.envs.gym_env import GymEnv

from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO

import csv
import joblib
import numpy as np
import os
import pickle
import tensorflow as tf
from math import pi

stub(globals())

file1 = 'data/local/posticml-trpo-maml-antLineDirect-200/maml11_fbs20_mbs40_flr_0.1_mlr0.01/itr_750.pkl'
#file1 = '../data/local/posticml-trpo-maml-antpos-200/maml1_fbs20_mbs40_flr_0.1_mlr0.01/itr_750.pkl'
#file2 = 'data/s3/posticml-trpo-maml-ant200/randenv100traj/itr_575.pkl'
#file3 = 'data/s3/posticml-trpo-maml-ant200/oracleenv100traj/itr_550.pkl'



make_video = True  # generate results if False, run code to make video if True
run_id = 7  # for if you want to run this script in multiple terminals (need to have different ids for each run)

if not make_video:
	test_num_goals = 40
	np.random.seed(1)
	goals = np.random.uniform(0.0, 3.0, size=(test_num_goals, ))
else:
	np.random.seed(1)
	test_num_goals = 1
	goals = [-pi]
	file_ext = 'mp4'  # can be mp4 or gif
print(goals)


gen_name = 'icml_ant_results_'
names = ['maml','pretrain','random', 'oracle']
exp_names = [gen_name + name for name in names]

step_sizes = [0.1]
#initial_params_files = [file1, file2, None, file3]

initial_params_files = [file1]

all_avg_returns = []
for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
	avg_returns = []

	for goal in goals:

		if initial_params_file is not None and 'oracle' in initial_params_file:
			env = normalize(AntEnvOracle())
			n_itr = 1
		else:
			env = normalize(AntEnvRandLineDirec())
			n_itr = 200
		env = TfEnv(env)
		policy = GaussianMLPPolicy(  # random policy
			name='policy',
			env_spec=env.spec,
			hidden_nonlinearity=tf.nn.relu,
			hidden_sizes=(100, 100),
		)

		if initial_params_file is not None:
			policy = None

		baseline = LinearFeatureBaseline(env_spec=env.spec)


		algo = TRPO(#VPG
			env=env,
			policy=policy,
			load_policy=initial_params_file,
			baseline=baseline,
			batch_size=4000,  # 2x
			max_path_length=5,
			n_itr=n_itr,
			reset_arg=goal,
			#optimizer_args={'init_learning_rate': step_sizes[step_i], 'tf_optimizer_args': {'learning_rate': 0.5*step_sizes[step_i]}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
		)
		""""
		algo = TRPO(
			env=env,
			policy=policy,
			baseline=baseline,
			batch_size=4000,
			max_path_length=env.horizon,
			n_itr=10000,
			discount=0.99,
			step_size=0.01,
			force_batch_sampler=True,  # for TF
			# Uncomment both lines (this and the plot parameter below) to enable plotting
			plot=True,
		)
		"""

		run_experiment_lite(
			algo.train(),
			# Number of parallel workers for sampling
			n_parallel=4,
			# Only keep the snapshot parameters for the last iteration
			snapshot_mode="all",
			# Specifies the seed for the experiment. If this is not provided, a random seed
			# will be used
			seed=1,
			exp_prefix='ant_test_posticml',
			exp_name='test' + str(run_id),
			plot=True,
		)





		# get return from the experiment
		with open('data/local/ant-test-posticml/test'+str(run_id)+'/progress.csv', 'r') as f:
			reader = csv.reader(f, delimiter=',')
			i = 0
			row = None
			returns = []
			for row in reader:
				i+=1
				if i ==1:
					ret_idx = row.index('AverageReturn')
				else:
					returns.append(float(row[ret_idx]))
			avg_returns.append(returns)

		if make_video:
			data_loc = 'data/local/ant-test-posticml/test'+str(run_id)+'/'
			save_loc = 'data/local/ant-test-posticml/test'+str(run_id)+'/'
			param_file = initial_params_file
			save_prefix = save_loc + names[step_i] + '_goal_' + str(goal)
			video_filename = save_prefix + 'prestep.' + file_ext
			os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=300 --video_filename='+video_filename)
			for itr_i in range(1,200):
				param_file = data_loc + 'itr_' + str(itr_i)  + '.pkl'
				video_filename = save_prefix + 'step_'+str(itr_i)+'.'+file_ext
				os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=300 --video_filename='+video_filename)




	all_avg_returns.append(avg_returns)



	task_avg_returns = []
	for itr in range(len(all_avg_returns[step_i][0])):
		task_avg_returns.append([ret[itr] for ret in all_avg_returns[step_i]])

	if not make_video:
		results = {'task_avg_returns': task_avg_returns}
		with open(exp_names[step_i] + '.pkl', 'wb') as f:
			pickle.dump(results, f)


for i in range(len(initial_params_files)):
	returns = []
	std_returns = []
	returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
	std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))
	print(initial_params_files[i])
	print(returns)
	print(std_returns)



