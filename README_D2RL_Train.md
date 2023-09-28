# Dense reinforcement learning agent training

# Introduction
This documentation mainly discusses the D2RL training process and the usage of the trained D2RL agent in the paper titled "Dense reinforcement learning for safety validation of autonomous vehicles".

# Installation
## Pre-requirements
  - Python installation
    - This repository is developed and tested under python 3.8.18 on Ubuntu 22.04 system.
  - Download all required datasets
    - **Need to be updated**
## Installation and configuration
### Clone this repository
```bash
git clone https://github.com/michigan-traffic-lab/Dense-Deep-Reinforcement-Learning.git
```
### Create a new conda virtual environment (Optional)
To ensure high flexibility, it is recommended to use a virtual environment when running this repository. To set up the virtual environment, please first [install Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html), then follow the commands provided below:
```bash
conda create -n d2rl python=3.8.18
```

```bash
conda activate d2rl
```

### Install all required packages
Due to the compatibility issue betwen gym and python setuptools, user should run follow commands to install a specific version of setuptools and wheel first:

```bash
pip install wheel==0.38.4
pip install setuptools==65.5.0
```

Then, to install the Python packages required for this repository, execute the command provided below:
```bash
pip install -r requirements_d2rl_train.txt
```
# Usage

> Please note for Step 1 and Step 4, the user should activate the Python environment installed in README.md as both of them are running testing experiments.

> For Step 2 and Step 3, the user should activate the Python environment installed in README_D2RL_Train.md as both of them are running training experiments.

## 1. AV Testing Using Behavior Policy
Please run the following commands to run the simulation and generate the raw experiment results for behavior policy-based testing (the experiment_name can be specified by users):
```bash
python main.py --experiment_name 2lane_400m_behavior_policy_testing --mode behavior_policy
```
By default, the simulation result will be stored in `./data_analysis/raw_data/your_experiment_name`.

Since the collection of AV testing data using behavior policy is also time-consuming, we also provide pre-collected data, which can be downloaded from [here](https://dense-deep-reinforcement-learning.s3.us-east-2.amazonaws.com/Experiment-2lane_400m_behavior_policy_testing_2023-09-03.zip). We recommend the user to create "./data_analysis/behavior_policy_data" subfolder to store the extracted dataset. 

## 2. D2RL agent training given behavior policy trajectories

Please run the following commands to train the D2RL agent:
```bash
python d2rl_train.py --yaml_conf ./d2rl_training/d2rl_train.yaml
```
The user will need to specify the training parameters in the `d2rl_train.yaml` file. The trained D2RL agent will be stored in `./ray_results/your_experiment_name`.
```yaml
experiment_name: "2lane_400m_D2RL_Training" # The name of the experiment
root_folder: "./data_analysis/behavior_policy_data/" # The folder that stores the behavior policy trajectories
data_folders: ["Experiment-2lane_400m_behavior_policy_testing_2023-09-03"] # The folder name of the behavior policy trajectories
data_folder_weights: [1]
local_dir: "./ray_results/" # The folder that stores the trained D2RL agent
num_workers: 12 # The number of workers used for training, please set it to the number of CPU cores
clip_reward_threshold: 100 # The reward clipping threshold
```
For the proposed D2RL training experiments, the episode reward will converge to around 80, which normally takes around 20-30 minutes.

## 3. Export the trained RLlib agent to a PyTorch model
Users will need to modify several arguments in the `rllib_model_export.py` file:
```python
checkpoint_path = "/media/mtl/2TB/Dense-Deep-Reinforcement-Learning/ray_results/2lane_400m_D2RL_Training_V2/PPO_my_env_959a3_00000_0_2023-09-03_16-29-30/checkpoint_000177/checkpoint-177" # replace with your rllib checkpoint path
export_dir = "./checkpoints/2lane_400m_D2RL" # replace with the path you would like to save the pytorch model
```

Please run the following commands to export the trained RLlib agent to a PyTorch model:
```bash
python rllib_model_export.py
```

## 4. Validate the trained D2RL agent

After the first three steps, users will be able to get the trained D2RL agent in the pytorch format. To validate the performance of the trained D2RL agent, users can refer to the "Usage" part in `README.md` to run the D2RL-based testing. However, the user will need to change the default pytorch model path to the path of the usertrained D2RL agent in line 34 of `main.py`. The default pytorch model path is `./checkpoints/2lane_400m_D2RL/`.
```python
d2rl_agent_path = "./checkpoints/2lane_400m_D2RL/model.pt"
```


# Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

# License

This project is licensed under the [PolyForm Noncommercial License 1.0.0]. Please refer to LICENSE for more details.

H. L. and the team have filed a US provisional patent application 63/338,424.

# Developers

Haowei Sun (haoweis@umich.edu)

Haojie Zhu (zhuhj@umich.edu)

Shuo Feng (fshuo@umich.edu)

For help or issues using the code, please create an issue for this repository or contact Haowei Sun (haoweis@umich.edu).

# Contact

For general questions about the paper, please contact Henry Liu (henryliu@umich.edu).
