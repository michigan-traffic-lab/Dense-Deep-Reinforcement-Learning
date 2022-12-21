from math import exp, isclose
from time import time
from mtlsp.logger.infoextractor import InfoExtractor
import copy
import json
import os
from functools import reduce
import conf.conf as conf
import numpy as np 
from pathlib import Path


class NADEInfoExtractor(InfoExtractor):
    def __init__(self, env):
        super().__init__(env)
        self.episode_log = {"collision_result": None, "collision_id": None, "weight_episode": 1, "current_weight": 1, 
                            "episode_info": None, "crash_decision_info": None, "decision_time_info": {}, "weight_step_info":{},
                            "drl_epsilon_step_info": {}, "real_epsilon_step_info": {}, "criticality_this_timestep": 0, "criticality_step_info": {}, "ndd_step_info": {}, "drl_obs_step_info":{}, "ttc_step_info":{}, "distance_step_info":{}, "av_obs":{}}
        self.record = {}
        self.initial_log = {}

    def add_initialization_info(self, vehID, information):
        """Add information about the initialization of the episode.

        Args:
            vehID (str): Vehicle ID.
            information (dict): Vehicle information including speed, postion, and lane index.
        """
        self.initial_log[vehID] = copy.deepcopy(information)

    def get_terminate_info(self, stop, reason, additional_info):
        """Obtain the information of the termination reason and all the vehicles in the simulation.

        Args:
            stop (bool): Whether the simulation is stopped.
            reason (dict): Stop reason flag.
            additional_info (dict): Collision vehicle ID.
        """
        # print("\033[32mweight_episode:\033[0m", self.episode_log["weight_episode"])
        if conf.experiment_config["mode"] == "DRL_train":
            return None
        if stop:
            self.episode_log["episode_info"] = self.env.episode_info
            if 1 in reason:  # have crash
                self.episode_log["collision_result"] = 1
                self.episode_log["collision_id"] = additional_info["collision_id"]
                crash_decision_info = {}
                all_vehicle_list = self.env.vehicle_list
                for id in self.episode_log["collision_id"]:
                    crash_decision_info[id] = all_vehicle_list[id].controller.ego_info
                    crash_decision_info[id]["action"] = all_vehicle_list[id].controller.action
                self.episode_log["crash_decision_info"] = crash_decision_info
            else:
                self.episode_log["collision_result"] = 0
            crash_log_flag = (1 in reason)
            all_log_flag = (conf.experiment_config["log_mode"] == "all")
            self.episode_log["initial_criticality"] = 1
            self.episode_log["initial_weight"] = self.env.initial_weight
            self.episode_log["reject_flag"] = False
            if 1:
            # if not (crash_log_flag or all_log_flag):  # if log all or crash event happens
                # will delete the most heavy part, only log the overall info
                self.episode_log["decision_time_info"] = None
                self.episode_log["crash_decision_info"] = None
            # self.delete_unused_epsilon_and_ndd_info()
            json_str = json.dumps(self.episode_log, indent=4)
            # print(self.save_dir)
            save_dir = os.path.join(self.save_dir, "tested_and_safe")
            if 1 in reason:
                save_dir = os.path.join(self.save_dir, "crash")
                with open(save_dir + "/"+str(self.episode_log["episode_info"]["id"]) + ".json", 'w') as json_file:
                    json_file.write(json_str)
                self.weight_result = float(self.episode_log["weight_episode"])
            #     # print("CRASH WEIGHT RESULT:", self.weight_result)
            else:
                save_dir = os.path.join(self.save_dir, "tested_and_safe")
                if self.meet_log_criteria(self.episode_log["ttc_step_info"], self.episode_log["distance_step_info"]):
                    with open(save_dir + "/"+str(self.episode_log["episode_info"]["id"]) + ".json", 'w') as json_file:
                        json_file.write(json_str)
                else:
                    file = os.path.join(self.env.simulator.experiment_path, "crash", self.env.simulator.output_filename+"."+"fcd.xml")
                    if os.path.isfile(file):
                        os.remove(file)
                self.weight_result = 0
            # total_criticality = sum(self.episode_log["criticality_step_info"].values())
            # if total_criticality > 0: # this episode have criticality
            #     with open(save_dir + "/"+str(self.episode_log["episode_info"]["id"]) + ".json", 'w') as json_file:
            #         json_file.write(json_str)

            if 6 in reason:
                save_dir = os.path.join(self.save_dir, "rejected")
            self.episode_log = {"collision_result": None, "collision_id": None, "weight_episode": 1,
                                "episode_info": None, "crash_decision_info": None, "decision_time_info": {}, "criticality_step_info":{}, "drl_obs_step_info":{}, "ttc_step_info":{}, "distance_step_info":{}, "av_obs":{}}
    
    def meet_log_criteria(self, ttc_dict, distance_dict):
        min_distance, min_ttc = self.calculate_min_distance_ttc(ttc_dict, distance_dict)
        ttc_flag = (min_ttc < 5)
        distance_flag = (min_distance < 10)
        return ttc_flag or distance_flag

    def calculate_min_distance_ttc(self, ttc_dict, distance_dict):
        min_ttc = 10000
        for timestep in ttc_dict:
            if ttc_dict[timestep] < min_ttc:
                min_ttc = ttc_dict[timestep]
        min_distance = 10000
        for timestep in distance_dict:
            if distance_dict[timestep] < min_distance:
                min_distance = distance_dict[timestep]
        return min_distance, min_ttc

    def get_current_ttc(self):
        return self.env.get_av_ttc()

    def get_criticality_this_step(self):
        return self.env.global_controller_instance_list[0].control_log["criticality"]
    
    def get_current_drl_obs(self):
        return self.env.global_controller_instance_list[0].control_log["discriminator_input"]

    # @profile
    def get_snapshot_info(self, control_info=None):
        """Obtain the vehicle information at every time step.
        """
        self.save_dir = self.env.simulator.experiment_path
        time_step = self.env.simulator.get_time()-self.env.simulator.step_size
        snapshot_weight_list = self.env.global_controller_instance_list[
            0].control_log["weight_list_per_simulation"]
        self.episode_log["weight_episode"] = self.episode_log["weight_episode"] * \
            reduce(lambda x, y: x * y, snapshot_weight_list)
        try:
            self.episode_log["distance_step_info"][time_step], self.episode_log["ttc_step_info"][time_step] = self.get_current_ttc()
        except:
            pass
        self.episode_log["current_weight"] = reduce(lambda x, y: x * y, snapshot_weight_list)
        self.episode_log["av_obs"][time_step] = self.env.get_av_obs()
        try:
            self.episode_log["criticality_step_info"][time_step] = self.get_criticality_this_step()
            if not isclose(self.episode_log["current_weight"], 1):
                self.episode_log["weight_step_info"][time_step] = self.episode_log["current_weight"]
                self.episode_log["drl_obs_step_info"][time_step] = self.get_current_drl_obs()
            if self.env.global_controller_instance_list[0].drl_epsilon_value != -1:
                self.episode_log["drl_epsilon_step_info"][time_step] = self.env.global_controller_instance_list[0].drl_epsilon_value
            if self.env.global_controller_instance_list[0].real_epsilon_value != -1:
                self.episode_log["real_epsilon_step_info"][time_step] = self.env.global_controller_instance_list[0].real_epsilon_value
            
            if "ndd_possi" in self.env.global_controller_instance_list[0].control_log:
                self.episode_log["ndd_step_info"][time_step] = self.env.global_controller_instance_list[0].control_log["ndd_possi"]
        except Exception as e:
            print("Log error:", e)
