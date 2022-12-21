from numpy import random
from numpy.core.numeric import full
from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
import numpy as np
from copy import deepcopy
import collections
import utils
from conf import conf
from controller.nddglobalcontroller import NDDBVGlobalController

class NADEBVGlobalController(NDDBVGlobalController):
    controlled_bv_num = 4

    def __init__(self, env, veh_type="BV"):
        super().__init__(env, veh_type)
        self.drl_info = None
        self.drl_epsilon_value = -1
        self.real_epsilon_value = -1


    # @profile
    def step(self):
        """Control the selected bvs from the bvs candidates to realize the decided behavior
        """
        self.real_epsilon_value = -1
        self.drl_epsilon_value = -1
        self.control_log = {"criticality":0, "discriminator_input":0}
        bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list = [], [], [], [], [], []
        vehicle_criticality_list = []
        self.reset_control_and_action_state()
        self.update_subscription(controller=TreeSearchNADEBackgroundController)

        # NDE decision
        for bv_id in self.controllable_veh_id_list:
            bv = self.env.vehicle_list[bv_id]
            bv.controller.step()

        # D2RL-trained intelligent testing environment (NADE) decision
        if self.apply_control_permission():
            if conf.experiment_config["mode"] == "NDE":
                for bv_id in self.controllable_veh_id_list:
                    bv = self.env.vehicle_list[bv_id]
                    if self.apply_control_permission():
                        bv.update()
            elif conf.experiment_config["mode"] == "D2RL":
                bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list, vehicle_criticality_list, _ = self.select_controlled_bv_and_action()
                for bv_id in self.controllable_veh_id_list:
                    bv = self.env.vehicle_list[bv_id]
                    if bv in controlled_bvs_list:
                        nade_action = bv_action_idx_list[controlled_bvs_list.index(bv)]
                        if nade_action is not None:
                            self.control_log["ndd_possi"] = ndd_possi_list[controlled_bvs_list.index(bv)]
                            bv.controller.action = utils.action_id_to_action_command(
                                nade_action)
                            bv.controller.NADE_flag = True
                            bv.simulator.set_vehicle_color(bv.id, bv.color_blue)
                    if self.apply_control_permission():
                        bv.update() # apply bv.controller.action
            else:
                raise ValueError("conf experiment mode not recognized. should be NDE or D2RL")
        self.control_log["weight_list_per_simulation"] = [
            val for val in weight_list if val is not None]
        if len(self.control_log["weight_list_per_simulation"]) == 0:
            self.control_log["weight_list_per_simulation"] = [1]
        return vehicle_criticality_list

    # @profile
    def select_controlled_bv_and_action(self):
        """Select the background vehicle controlled by D2RL-trained intelligent testing environment (NADE) and the corresponding action.

        Returns:
            list(float): List of action index for all studied background vehicles. 
            list(float): List of weight of each vehicle.
            float: Maximum criticality.
            list(float): List of behavior probability based on NDD.
            list(float): List of critical possibility.
            list(Vehicle): List of all studied vehicles.
        """
        num_controlled_critical_bvs = 1
        controlled_bvs_list = self.get_bv_candidates()
        CAV_obs = self.env.vehicle_list["CAV"].observation.information
        full_obs = self.get_full_obs_from_cav_obs_and_bv_list(CAV_obs, controlled_bvs_list)
        self.nade_candidates = controlled_bvs_list
        bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list = self.calculate_criticality_list(controlled_bvs_list, CAV_obs, full_obs)
        whole_weight_list = []
        self.control_log["criticality"] = sum(bv_criticality_list)
    
        discriminator_input = self.collect_discriminator_input_simplified(full_obs, controlled_bvs_list, bv_criticality_list) # get D2RL agent observation
        self.control_log["discriminator_input"] = discriminator_input.tolist()
        self.epsilon_value = -1
        underline_drl_action = self.get_underline_drl_action(discriminator_input, bv_criticality_list)
        
        if sum(bv_criticality_list) > 0:
            self.drl_epsilon_value = underline_drl_action
            self.real_epsilon_value = underline_drl_action

        for i in range(len(controlled_bvs_list)):
            bv = controlled_bvs_list[i]
            bv_criticality = bv_criticality_list[i]
            bv_criticality_array = criticality_array_list[i]
            bv_pdf = bv.controller.get_NDD_possi()
            combined_bv_criticality_array = bv_criticality_array
            bv_action_idx, weight, ndd_possi, critical_possi, single_weight_list = bv.controller.Decompose_sample_action(np.sum(combined_bv_criticality_array), combined_bv_criticality_array, bv_pdf, underline_drl_action)
            if bv_action_idx is not None:
                bv_action_idx = bv_action_idx.item()
            bv_action_idx_list.append(bv_action_idx), weight_list.append(weight), ndd_possi_list.append(ndd_possi), IS_possi_list.append(critical_possi)
            if single_weight_list is not None:
                whole_weight_list.append(min(single_weight_list))
            else:
                whole_weight_list.append(None)
                
        vehicle_criticality_list = deepcopy(bv_criticality_list)
        # Select the Principal Other Vehicle (POV) with highest criticality
        selected_bv_idx = sorted(range(len(bv_criticality_list)),
                                 key=lambda i: bv_criticality_list[i])[-num_controlled_critical_bvs:]
        for i in range(len(controlled_bvs_list)):
            if i in selected_bv_idx:
                if whole_weight_list[i] and whole_weight_list[i]*self.env.info_extractor.episode_log["weight_episode"]*self.env.initial_weight < conf.weight_threshold:
                    bv_action_idx_list[i], weight_list[i], ndd_possi_list[i], IS_possi_list[i] = None, None, None, None
            if i not in selected_bv_idx:
                bv_action_idx_list[i], weight_list[i], ndd_possi_list[i], IS_possi_list[i] = None, None, None, None
        if len(bv_criticality_list):
            max_vehicle_criticality = np.max(bv_criticality_list)
        else:
            max_vehicle_criticality = -np.inf

        return bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list, vehicle_criticality_list, discriminator_input

    def apply_control_permission(self):
        for vehicle in self.get_bv_candidates():
            if vehicle.controller.NADE_flag and utils.is_lane_change(vehicle.observation.information["Ego"]):
                return False
        return True
        
    # @profile
    def get_bv_candidates(self):
        """Find the Principal Other Vehicle (POV) candidates around the av

        Returns:
            list(Vehicle): List of background vehicles around the CAV.
        """
        av = self.env.vehicle_list["CAV"]
        av_pos = av.observation.local[av.id][66]
        av_context = av.observation.context
        av_context_bv_list = list(av_context.keys())
        bv_candidates = []
        bv_list = []
        # collect all bvs in a certain range of AV
        for bv_id in av_context_bv_list:
            bv_pos = av_context[bv_id][66]
            dist = utils.cal_euclidean_dist(av_pos, bv_pos)
            if dist <= conf.cav_obs_range:
                bv_list.append([bv_id, dist])
        # sort the bvs list by distance, and select the top controlled_bv_num nearest bvs
        bv_list.sort(key=lambda i: i[1])
        for i in range(len(bv_list)):
            if i < self.controlled_bv_num:
                bv_id = bv_list[i][0]
                bv = self.env.vehicle_list[bv_id]
                bv_candidates.append(bv)
        return bv_candidates

    @staticmethod
    # @profile
    def pre_load_predicted_obs_and_traj(full_obs):
        predicted_obs = {}
        trajectory_obs = {}
        action_list = ["left", "right", "still"]
        for veh_id in full_obs:
            for action in action_list:
                vehicle = full_obs[veh_id]
                if veh_id not in trajectory_obs:
                    trajectory_obs[veh_id] = {}
                if veh_id not in predicted_obs:
                    predicted_obs[veh_id] = {}
                predicted_obs[veh_id][action], trajectory_obs[veh_id][action] = TreeSearchNADEBackgroundController.update_single_vehicle_obs(vehicle, action)
        return predicted_obs, trajectory_obs
    
    def collect_discriminator_input_simplified(self, full_obs, controlled_bvs_list, bv_criticality_list):
        """D2RL agent observation collection
        """
        CAV_global_position = list(full_obs["CAV"]["position"])
        CAV_speed = full_obs["CAV"]["velocity"]
        tmp_weight = self.env.info_extractor.episode_log["weight_episode"]
        tmp_weight = np.log10(tmp_weight)
        vehicle_info_list = []
        controlled_bv_num = 1
        total_bv_info_length = controlled_bv_num * 4
        # print(bv_criticality_list)
        if len(bv_criticality_list):
            selected_bv_index = np.argmax(np.array(bv_criticality_list))
            vehicle = controlled_bvs_list[selected_bv_index]
            veh_id = vehicle.id
            vehicle_single_obs = full_obs[veh_id]
            vehicle_local_position = list(vehicle_single_obs["position"])
            vehicle_relative_position = [vehicle_local_position[0]-CAV_global_position[0], vehicle_local_position[1]-CAV_global_position[1]]
            vehicle_relative_speed = vehicle_single_obs["velocity"] - CAV_speed
            predict_relative_position = vehicle_relative_position[0] + vehicle_relative_speed
            vehicle_info_list.extend(vehicle_relative_position +[vehicle_relative_speed] + [predict_relative_position])
        else:
            vehicle_info_list.extend([-20, -8, -10, -20]) # fill the state space with default values
        if len(vehicle_info_list) < total_bv_info_length:
            vehicle_info_list.extend([-1]*(total_bv_info_length - len(vehicle_info_list)))
        bv_criticality_flag = (sum(bv_criticality_list) > 0)
        if sum(bv_criticality_list) > 0:
            bv_criticality_value = np.log10(sum(bv_criticality_list))
        else:
            bv_criticality_value = 16
        if conf.simulation_config["map"] == "2LaneLong": # 2lane 4000m experiment
            CAV_position_lb, CAV_position_ub = [400, 40], [4400, 50]
        else: # 2lane 400m experiment
            CAV_position_lb, CAV_position_ub = [400, 40], [800, 50]
        CAV_velocity_lb, CAV_velocity_ub = 0, 20
        weight_lb = -30
        weight_ub = 0
        bv_criticality_flag_lb = 0
        bv_criticality_flag_ub = 1
        bv_criticality_value_lb = -16
        bv_criticality_value_ub = 0
        vehicle_info_lb, vehicle_info_ub = [-20, -8, -10, -20], [20, 8, 10, 20]
        lb_array = np.array(CAV_position_lb + [CAV_velocity_lb] + [weight_lb] + [bv_criticality_flag_lb] + [bv_criticality_value_lb] + vehicle_info_lb * controlled_bv_num)
        ub_array = np.array(CAV_position_ub + [CAV_velocity_ub] + [weight_ub] + [bv_criticality_flag_ub] + [bv_criticality_value_ub] + vehicle_info_ub * controlled_bv_num)
        total_obs_for_DRL_ori = np.array(CAV_global_position + [CAV_speed] + [tmp_weight] + [bv_criticality_flag] + [bv_criticality_value] + vehicle_info_list)
        total_obs_for_DRL = 2 * (total_obs_for_DRL_ori - lb_array)/(ub_array - lb_array) - 1 # normalize the observation
        total_obs_for_DRL = np.clip(total_obs_for_DRL, -5, 5) # clip the observation
        return np.float32(np.array(total_obs_for_DRL))

    def calculate_criticality_list(self, controlled_bvs_list, CAV_obs, full_obs):
        bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list = [], [], [], [], [], []        
        predicted_full_obs, predicted_traj_obs = NADEBVGlobalController.pre_load_predicted_obs_and_traj(full_obs)
        CAV_left_prob, CAV_still_prob, CAV_right_prob = NADEBVGlobalController._get_Surrogate_CAV_action_probability(cav_obs=self.env.vehicle_list["CAV"].observation.information)
        for bv in controlled_bvs_list:
            bv_criticality, criticality_array = bv.controller.Decompose_decision(
                CAV_obs, SM_LC_prob=[CAV_left_prob, CAV_still_prob, CAV_right_prob], full_obs=full_obs, predicted_full_obs=predicted_full_obs, predicted_traj_obs=predicted_traj_obs)
            bv_criticality_list.append(bv_criticality), criticality_array_list.append(criticality_array)
        return bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list


    def get_underline_drl_action(self, discriminator_input, bv_criticality_list):
        underline_drl_action = None # 1 - adversarial maneuver probability
        if sum(bv_criticality_list) > 0:
            # critical time step
            if conf.simulation_config["epsilon_setting"] == "drl": # using D2RL agent to output the adversarial maneuver probability
                if conf.discriminator_agent is None:
                    conf.discriminator_agent = conf.load_discriminator_agent()
                underline_drl_action = conf.discriminator_agent.compute_action(discriminator_input)
                if sum(bv_criticality_list) > 0:
                    print(underline_drl_action, self.env.info_extractor.episode_log["weight_episode"])
                underline_drl_action = max(0, min(underline_drl_action, 1))
            elif conf.simulation_config["epsilon_setting"] == "fixed": # ! need to be corrected
                underline_drl_action = conf.epsilon_value
                underline_drl_action = conf.epsilon_value
        return underline_drl_action

    @staticmethod
    # @profile
    def _get_Surrogate_CAV_action_probability(cav_obs):
        """Predict the action probability of the CAV based on surrogate model"""
        CAV_left_prob, CAV_right_prob = 0, 0
        CAV_still_prob = conf.epsilon_still_prob
        left_gain, right_gain = 0, 0
        left_LC_safety_flag, right_LC_safety_flag = False, False
        lane_index_list = [-1, 1]  # -1: right turn; 1: left turn
        for lane_index in lane_index_list:
            LC_safety_flag, gain = NADEBVGlobalController._Mobil_surraget_model(
                cav_obs=cav_obs, lane_index=lane_index)
            if gain is not None:
                if lane_index == -1:
                    right_gain = np.clip(gain, 0., None)
                    right_LC_safety_flag = LC_safety_flag
                elif lane_index == 1:
                    left_gain = np.clip(gain, 0., None)
                    left_LC_safety_flag = LC_safety_flag
        assert(left_gain >= 0 and right_gain >= 0)

        if not cav_obs["Ego"]["could_drive_adjacent_lane_left"]:
            left_LC_safety_flag = 0
            left_gain = 0
        elif not cav_obs["Ego"]["could_drive_adjacent_lane_right"] == 0:
            right_LC_safety_flag = 0
            right_gain = 0

        CAV_left_prob += conf.epsilon_lane_change_prob*left_LC_safety_flag
        CAV_right_prob += conf.epsilon_lane_change_prob*right_LC_safety_flag

        max_remaining_LC_prob = 1-conf.epsilon_still_prob-CAV_left_prob-CAV_right_prob

        total_gain = left_gain+right_gain
        obtained_LC_prob_for_sharing = np.clip(utils.remap(total_gain, [0, conf.SM_MOBIL_max_gain_threshold], [
                                               0, max_remaining_LC_prob]), 0, max_remaining_LC_prob)
        CAV_still_prob += (max_remaining_LC_prob -
                           obtained_LC_prob_for_sharing)

        if total_gain > 0:
            CAV_left_prob += obtained_LC_prob_for_sharing * \
                (left_gain/(left_gain + right_gain))
            CAV_right_prob += obtained_LC_prob_for_sharing * \
                (right_gain/(left_gain + right_gain))

        assert(0.99999 <= (CAV_left_prob + CAV_still_prob + CAV_right_prob) <= 1.0001)
        return CAV_left_prob, CAV_still_prob, CAV_right_prob

    @staticmethod
    # @profile
    def _Mobil_surraget_model(cav_obs, lane_index):
        """Apply the Mobil surrogate model for CAV Lane change to calculate the gain for this lane change maneuver. If it does not have safety issue, then return True, gain; otherwise False, None.

        Args:
            lane_index (integer): Candidate lane for the change.

        Returns:
            (bool, float): The first output stands for safety flag (whether ADS will crash immediately after doing LC), the second output is gain (Now could even smaller than 0).
        """
        gain = None
        cav_info = cav_obs['Ego']

        if lane_index == -1:  # right turn
            new_preceding = cav_obs["RightLead"]
            new_following = cav_obs["RightFoll"]
        if lane_index == 1:  # left turn
            new_preceding = cav_obs["LeftLead"]
            new_following = cav_obs["LeftFoll"]

        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding["distance"]
        if new_following:
            r_new_following = new_following["distance"]
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=cav_info)

        old_preceding = cav_obs["Lead"]
        old_following = cav_obs["Foll"]
        self_pred_a = utils.acceleration(
            ego_vehicle=cav_info, front_vehicle=new_preceding)

        if new_following_pred_a < -conf.Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return True, 0

        # calculate acceleration advantage
        self_a = utils.acceleration(
            ego_vehicle=cav_info, front_vehicle=old_preceding)
        old_following_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=cav_info)
        old_following_pred_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + conf.Surrogate_POLITENESS * \
            (new_following_pred_a - new_following_a +
             old_following_pred_a - old_following_a)
        return True, gain

    # @profile
    def get_full_obs_from_cav_obs_and_bv_list(self, CAV_obs, bv_list):
        # This observation will be a dict containing CAV and all BV candidates
        full_obs = collections.OrderedDict()
        full_obs["CAV"] = CAV_obs["Ego"]
        vehicle_id_list = [vehicle.id for vehicle in bv_list]
        cav_surrounding = self._process_cav_context(vehicle_id_list)
        av_pos = CAV_obs["Ego"]["position"]
        for vehicle in bv_list:
            vehicle_id = vehicle.observation.information["Ego"]["veh_id"]
            full_obs[vehicle_id] = vehicle.observation.information["Ego"]
            bv_pos = vehicle.observation.information["Ego"]["position"]
        return full_obs

    # @profile
    def _process_cav_context(self, vehicle_id_list):
        """fetch information of all bvs from the cav context information
        """
        cav = self.env.vehicle_list["CAV"]
        cav_pos = cav.observation.local["CAV"][66]
        cav_context = cav.observation.context
        cav_surrounding = {}
        cav_surrounding["CAV"] = {
            "range": 0,
            "lane_width": self.env.simulator.get_vehicle_lane_width("CAV"),
            "lateral_offset": cav.observation.local["CAV"][184],
            "lateral_speed": cav.observation.local["CAV"][50],
            "position": cav_pos,
            "prev_action": cav.observation.information["Ego"]["prev_action"],
            "relative_lane_index": 0,
            "speed": cav.observation.local["CAV"][64]
        }
        total_vehicle_id_list = list(
            set(vehicle_id_list) | set(cav_context.keys()))
        for veh_id in total_vehicle_id_list:
            bv_pos = cav_context[veh_id][66]
            distance = self.env.simulator.get_vehicles_dist_road("CAV", veh_id)

            if distance > conf.cav_obs_range+5:
                distance_alter = self.env.simulator.get_vehicles_dist_road(
                    veh_id, "CAV")
                if distance_alter > conf.cav_obs_range+5:
                    continue
                else:
                    distance = -distance_alter
                    relative_lane_index = - \
                        self.env.simulator.get_vehicles_relative_lane_index(
                            veh_id, "CAV")
            else:
                relative_lane_index = self.env.simulator.get_vehicles_relative_lane_index(
                    "CAV", veh_id)
            cav_surrounding[veh_id] = {
                "range": distance,
                "lane_width": self.env.simulator.get_vehicle_lane_width(veh_id),
                "lateral_offset": cav_context[veh_id][184],
                "lateral_speed": cav_context[veh_id][50],
                "position": bv_pos,
                "prev_action": self.env.vehicle_list[veh_id].observation.information["Ego"]["prev_action"],
                "relative_lane_index": relative_lane_index,
                "speed": cav_context[veh_id][64]
            }
        return cav_surrounding
