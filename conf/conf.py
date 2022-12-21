from math import exp
from .defaultconf import *
import torch
weight_threshold = 0
epsilon_value = 0.99
simulation_config["map"] = "2Lane" # simulation map definition
d2rl_agent_path = "./checkpoints/2lane_400m/model.pt" # the pytorch checkpoint of the D2RL agent

simulation_config["epsilon_type"] = "continious" # define whether the d2rl agent will output continious/discrete adversarial action probability
experiment_config["AV_model"] = "IDM" # Tested AV models
simulation_config["speed_mode"] = "high_speed" # the speed profile of the vehicles in the simulation
simulation_config["gui_flag"] = False # whether to show the simulation in GUI

discriminator_agent = None
experiment_config["root_folder"] = "./data_analysis/raw_data" # the folder to save the simulation data
experiment_config["episode_num"] = 1232

# tree search-based maneuver challenge calculation configuration
treesearch_config["search_depth"] = 1
treesearch_config["surrogate_model"] = "AVI" # "AVI" "surrogate"
treesearch_config["offline_leaf_evaluation"] = False
treesearch_config["offline_discount_factor"] = 1
treesearch_config["treesearch_discount_factor"] = 1

simulation_config["initialization_rejection_sampling_flag"] = False
experiment_config["log_mode"] = "crash" # "all" "crash"
traffic_flow_config["BV"] = True
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 8  # [m/s2]

# D2RL-based agent definition
class torch_discriminator_agent:
    def __init__(self, checkpoint_path):
        if not checkpoint_path:
            checkpoint_path = "./model.pt"
        print("Loading checkpoint", checkpoint_path)
        self.model = torch.jit.load(checkpoint_path)
        self.model.eval()

    def compute_action(self, observation):
        lb = 0.001
        ub = 0.999
        obs = torch.reshape(torch.tensor(observation), (1,len(observation)))
        out = self.model({"obs":obs},[torch.tensor([0.0])],torch.tensor([1]))
        if simulation_config["epsilon_type"] == "discrete":
            action = torch.argmax(out[0][0])
        else:
            action = np.clip((float(out[0][0][0])+1)*(ub-lb)/2 + lb, lb, ub)
        return action

# load pytorch checkpoints into the D2RL agent
def load_discriminator_agent(mode="torch", checkpoint_path=d2rl_agent_path): #mode="ray"
    if mode == "ray":
        import ray
        import ray.rllib.agents.ppo as ppo
        from ray.tune.registry import register_env
        from gym_env import DRL_gym_ENV
        from gym_env_offline import DRL_gym_ENV_offline
        def env_creator(env_config):
            return DRL_gym_ENV_offline()  # return an env instance
        register_env("my_env", env_creator)
        if experiment_config["mode"] == "DRL_train":
            ray.init(address=os.environ["ip_head"], include_dashboard=False, ignore_reinit_error=True)
        else:
            ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = 0
        config["num_workers"] = 0
        config["framework"] = "torch"
        config["explore"] = False
        discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
        if not checkpoint_path:
            checkpoint_path = "/gpfs/accounts/henryliu_root/henryliu1/shared_data/ray_results/2Lane_400m_099_DRL_behavior_policy_train_new_DRL/PPO_my_env_4206c_00000_0_2022-03-12_09-11-06/checkpoint_51/checkpoint-51"
        discriminator_agent.restore(checkpoint_path)
    elif mode == "torch":
        discriminator_agent = torch_discriminator_agent(checkpoint_path)
    else:
        raise NotImplementedError("unsupported mode in discriminator agent load!")
    return discriminator_agent