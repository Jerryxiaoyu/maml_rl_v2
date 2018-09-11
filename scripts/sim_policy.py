import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

''''
usage

python scripts/sim_policy.py   data/local/ant-test-posticml/test1/itr_2.pkl   --speedup=4 --max_path_length=300 --video_filename=data/local/ant-test-posticml/test1/prestep.mp4

python scripts/sim_policy.py   data/local/posticml-trpo-maml-antLineDirect-200/maml1_fbs20_mbs40_flr_0.1_mlr0.01/itr_675.pkl   --speedup=4 --max_path_length=300 --video_filename=data/local/ant-test-posticml/test1/prestep.mp4
'''''
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    args = parser.parse_args()

    max_tries = 10
    tri = 0
    while True:
        tri += 1
        with tf.Session() as sess:
            data = joblib.load(args.file)
            policy = data['policy']
            env = data['env']
            while True:
                path = rollout(env, policy, max_path_length=args.max_path_length,
                               animated=True, speedup=args.speedup, video_filename=args.video_filename)
                if args.prompt:
                    if not query_yes_no('Continue simulation?'):
                        break
                else:
                    break
            #import pdb; pdb.set_trace()
        if len(path['rewards']) < args.max_path_length and tri >= max_tries:
            tf.reset_default_graph()
            continue
        break
