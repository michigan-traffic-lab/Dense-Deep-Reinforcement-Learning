from ray.tune.trial import ExportFormat
import torch
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from d2rl_training.d2rl_training_env import D2RLTrainingEnv

import yaml, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_conf', type=str, default='d2rl_training/d2rl_train.yaml', metavar='N', help='the yaml configuration file path')
args = parser.parse_args()
try:
    with open(args.yaml_conf, 'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print("Yaml configuration file not successfully loaded:", e)


def env_creator(env_config):
    return D2RLTrainingEnv(yaml_conf)
register_env("my_env", env_creator)

ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 0
config["framework"] = "torch"
config["explore"] = False
discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
checkpoint_path = "/media/mtl/2TB/Dense-Deep-Reinforcement-Learning/ray_results/2lane_400m_D2RL_Training_V2/PPO_my_env_959a3_00000_0_2023-09-03_16-29-30/checkpoint_000177/checkpoint-177" # replace with your checkpoint path
discriminator_agent.restore(checkpoint_path)
p = discriminator_agent.get_policy()
export_dir = "./checkpoints/2lane_400m_D2RL" # replace with the path you would like to save the pytorch model
p.export_model(export_dir) 

model = torch.jit.load(export_dir + "/model.pt") 
model.eval()

def compute_action_torch(observation):
    obs = torch.reshape(torch.tensor(observation), (1,len(observation)))
    out = model({"obs":obs},[torch.tensor([0.0])],torch.tensor([1]))
    action = torch.argmax(out[0][0])
    return action

def compute_action_torch_continuous(observation):
    obs = torch.reshape(torch.tensor(observation), (1,len(observation)))
    out = model({"obs":obs},[torch.tensor([0.0])],torch.tensor([1]))
    # print(out)
    action = np.clip((float(out[0][0][0])+1)*(0.999-0.001)/2 + 0.001, 0.001, 0.999)
    return action

import gym
env = D2RLTrainingEnv(yaml_conf)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        ray_action = discriminator_agent.compute_action(observation)
        torch_action = compute_action_torch_continuous(observation)
        print(f"Ray action: {ray_action}. Torch action: {torch_action}")
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()