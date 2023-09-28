import ray
from ray import tune
import glob
from ray.tune.registry import register_env
from d2rl_training.d2rl_training_env import D2RLTrainingEnv
import yaml, argparse
from tqdm import tqdm
import json
parser = argparse.ArgumentParser()
parser.add_argument('--yaml_conf', type=str, default='./d2rl_training/d2rl_train.yaml', metavar='N', help='the yaml configuration file path')
args = parser.parse_args()

try:
    with open(args.yaml_conf, 'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print("Yaml configuration file not successfully loaded:", e)


def env_creator(env_config):
    
    return D2RLTrainingEnv(yaml_conf)
register_env("my_env", env_creator)
ray.init(include_dashboard=False, ignore_reinit_error=True)


import os
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import AgentID, PolicyID
if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        episode.hist_data["constant"] = []
        episode.hist_data["weight_reward"] = []
        episode.hist_data["exposure"] = []
        episode.hist_data["positive_weight_reward"] = []
        episode.hist_data["episode_num"] = []
        episode.hist_data["step_num"] = []

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        last_info = episode.last_info_for()
        # for key in episode.hist_data:
        #     episode.hist_data[key].append(last_info[key])
        # print(last_info)

def calculate_crash_weight(root_folder, path_list):
    for path in path_list:
        crash_weight_dict = {}
        data_folder = root_folder + path
        crash_path = os.path.join(data_folder, "crash")
        print(crash_path)
        if os.path.isdir(crash_path):
            crash_data_path_list = glob.glob(crash_path+"/*.json")
        else:
            crash_data_path_list = None
            print("Crash data path not found!")
        crash_weight_list = []
        for crash_data_path in tqdm(crash_data_path_list):
            with open(crash_data_path) as crash_data:
                crash_data_json = json.load(crash_data)
                crash_weight_list.append(crash_data_json["weight_episode"])
                if crash_data_json["weight_episode"] < 0.1:
                    crash_weight_dict[crash_data_path] = [crash_data_json["weight_episode"],crash_data_json["weight_episode"]]
        json_str = json.dumps(crash_weight_dict, indent=4)
        with open(data_folder + "/" + "crash_weight_dict.json", "w") as json_file:
            json_file.write(json_str)


calculate_crash_weight(yaml_conf["root_folder"], yaml_conf["data_folders"])
print("Nodes in the Ray cluster:")
print(ray.nodes())
tune.run(
    "PPO",
    stop={"training_iteration": 1000},
    config={
        "env": "my_env",
        "num_gpus": 0,
        "num_workers": yaml_conf["num_workers"],
        "num_envs_per_worker": 1,
        "gamma": 1.0,
        "rollout_fragment_length": 600,
        "vf_clip_param": yaml_conf["clip_reward_threshold"],
        "framework": "torch",
        "ignore_worker_failures": True,
        "callbacks": MyCallbacks,
    },
    checkpoint_freq=1,
    local_dir=yaml_conf["local_dir"],
    name=yaml_conf["experiment_name"],
)