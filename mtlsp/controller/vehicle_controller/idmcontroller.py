from __future__ import division, print_function
import numpy as np
from mtlsp.simulator import Simulator
from mtlsp.controller.vehicle_controller.controller import Controller, BaseController


# Longitudinal policy parameters
CAV_COMFORT_ACC_MAX = 1.5 #2.0  # [m/s2]
CAV_COMFORT_ACC_MIN = -2 #-4.0  # [m/s2]
CAV_DISTANCE_WANTED = 2.0 #5.0  # [m]
CAV_TIME_WANTED = 1.2 #1.5  # [s]
CAV_DESIRED_VELOCITY = 33.33 # 33.33 #35 # [m/s]
CAV_DELTA = 4.0  # []

# Lateral policy parameters
CAV_POLITENESS = 0.  # in [0, 1]
CAV_LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
CAV_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
CAV_LANE_CHANGE_DELAY = 1.0  # [s]
# NDD Vehicle IDM parameters
COMFORT_ACC_MAX = 2 # [m/s2]
COMFORT_ACC_MIN = -4.0  # [m/s2]
DISTANCE_WANTED = 5.0  # [m]
TIME_WANTED = 1.5  # [s]
DESIRED_VELOCITY = 35 # [m/s]
DELTA = 4.0  # []
# CAV surrogate model IDM parameter
SM_IDM_COMFORT_ACC_MAX = 2.0  # [m/s2]  2
SM_IDM_COMFORT_ACC_MIN = -4.0  # [m/s2]  -4
SM_IDM_DISTANCE_WANTED = 5.0  # [m]  5
SM_IDM_TIME_WANTED = 1.5  # [s]  1.5
SM_IDM_DESIRED_VELOCITY = 35 # [m/s]
SM_IDM_DELTA = 4.0  # []

acc_low = -4
acc_high = 2
LENGTH = 5

def acceleration(ego_vehicle=None, front_vehicle=None, mode=None):
    """Compute an acceleration command with the Intelligent Driver Model. The acceleration is chosen so as to:
        - reach a target velocity;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

    Args:
        ego_vehicle (dict, optional): Information of the vehicle whose desired acceleration is to be computed. It does not have to be an IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to reason about other vehicles behaviors even though they may not IDMs. Defaults to None.
        front_vehicle (dict, optional): Information of the vehicle preceding the ego-vehicle. Defaults to None.
        mode (str, optional): Difference IDM parameters for BV and CAV. Defaults to None.

    Returns:
        float: Acceleration command for the ego-vehicle in m/s^2.
    """
    if not ego_vehicle:
        return 0

    if mode == "CAV":
        a0 = CAV_COMFORT_ACC_MAX
        v0 = CAV_DESIRED_VELOCITY
        delt = CAV_DELTA
    else:
        a0 = COMFORT_ACC_MAX
        v0 = SM_IDM_DESIRED_VELOCITY
        delt = SM_IDM_DELTA
    acceleration = a0 * \
        (1 - np.power(ego_vehicle["velocity"] /
                      v0, delt))
    if front_vehicle is not None:
        r = front_vehicle["distance"]
        d = max(1e-5, r - LENGTH)
        acceleration -= a0 * \
            np.power(desired_gap(ego_vehicle, front_vehicle, mode) / d, 2)
    return acceleration

def desired_gap(ego_vehicle, front_vehicle=None, mode=None):
    """Compute the desired distance between a vehicle and its leading vehicle.

    Args:
        ego_vehicle (dict): Information of the controlled vehicle.
        front_vehicle (dict, optional): Information of the leading vehicle. Defaults to None.
        mode (str, optional): Difference IDM parameters for BV and CAV. Defaults to None.

    Returns:
        float: Desired distance between the two vehicles in m.
    """
    if mode == "CAV":
        d0 = CAV_DISTANCE_WANTED
        tau = CAV_TIME_WANTED
        ab = -CAV_COMFORT_ACC_MAX * CAV_COMFORT_ACC_MIN
    else:
        d0 = DISTANCE_WANTED
        tau = TIME_WANTED
        ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN
    dv = ego_vehicle["velocity"] - front_vehicle["velocity"]
    d_star = d0 + max(0, ego_vehicle["velocity"] * tau +
                      ego_vehicle["velocity"] * dv / (2 * np.sqrt(ab)))
    return d_star


class IDMController(BaseController):
    """A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """
    def __init__(self, subscription_method=Simulator.subscribe_vehicle_all_information, controllertype="IDMController"):
        super().__init__(subscription_method=subscription_method, controllertype=controllertype)
        self.controller_parameter = {}
        self.mode = None

    def install(self):
        super().install()
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_red)

    @staticmethod
    def decision_pdf(observation):
        """Generate decision probability distribution.

        Args:
            observation (Observation): Vehicle observation.

        Returns:
            list(float): Maneuver probability distribution.
        """        
        action, mode = IDMController.decision(observation)
        pdf = [0,0,0]
        if action["lateral"] == "left":
            pdf[0] = 1
        elif action["lateral"] == "right":
            pdf[2] = 1
        else:
            pdf[1] = 1
        return pdf
    
    @staticmethod
    def decision(observation):
        """Vehicle decides next action based on IDM model.

        Args:
            observation (dict): Observation of the vehicle.

        Returns:
            float: Action index. 0 represents left turn and 1 represents right turn. If the output is larger than 2, it is equal to the longitudinal acceleration plus 6.
        """
        action = None
        mode = None
        ego_vehicle = observation["Ego"]
        front_vehicle = observation["Lead"]       
        # Lateral: MOBIL
        mode = "MOBIL"
        left_gain, right_gain = 0, 0
        left_LC_flag, right_LC_flag = False, False  
        current_lane_id = observation["Ego"]["lane_index"]
        possible_lane_change = [-1,1]
        if not observation["Ego"]["could_drive_adjacent_lane_right"]:
            possible_lane_change = [1]
        elif not observation["Ego"]["could_drive_adjacent_lane_left"]:
            possible_lane_change = [-1]

        for lane_index in possible_lane_change:
            LC_flag, gain = IDMController.mobil_gain(lane_index, observation)
            if LC_flag and gain:
                if lane_index < 0: 
                    right_gain, right_LC_flag = np.clip(gain, 0., None), LC_flag
                elif lane_index > 0: 
                    left_gain, left_LC_flag = np.clip(gain, 0., None), LC_flag
        if left_LC_flag or right_LC_flag:
            if right_gain > left_gain:
                action = {
                    "lateral": "right",
                    "longitudinal": 0
                }
                assert(right_gain >= CAV_LANE_CHANGE_MIN_ACC_GAIN)
            else:
                action = {
                    "lateral": "left",
                    "longitudinal": 0
                }
                assert(left_gain >= CAV_LANE_CHANGE_MIN_ACC_GAIN)
        # Longitudinal: IDM
        else:
            mode = "IDM"
            if "AV" in ego_vehicle["veh_id"]:
                veh_type = "CAV"
            else:
                veh_type = "BV"
            tmp_acc = acceleration(ego_vehicle=ego_vehicle,front_vehicle=front_vehicle,mode=veh_type)
            tmp_acc = np.clip(tmp_acc, acc_low, acc_high)
            action = {
                "lateral": "central",
                "longitudinal": tmp_acc
            }
        return action, mode

    @staticmethod
    def mobil_gain(lane_index, observation):
        """MOBIL model for the vehicle.

        Args:
            lane_index (int): Lane change direction, -1 represents right, 1 represents left.
            observation (dict): Observation of the vehicle.

        Returns:
            bool, float: Whether it is feasible to do the lane change maneuver. Lane change gain.
        """
        gain = None
        ego = observation["Ego"]
        if "AV" in ego["veh_id"]:
            mode = "CAV"
        else:
            mode = "BV"        

        # Is the maneuver unsafe for the new following vehicle?
        if lane_index == 1:
            new_preceding, new_following = observation["LeftLead"], observation["LeftFoll"]
        if lane_index == -1:
            new_preceding, new_following = observation["RightLead"], observation["RightFoll"]
        
        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding["distance"]
        if new_following:
            r_new_following = new_following["distance"]       
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = acceleration(ego_vehicle=new_following, front_vehicle=new_preceding, mode=mode)
        new_following_pred_a = acceleration(ego_vehicle=new_following, front_vehicle=ego, mode=mode)

        old_preceding, old_following = observation["Lead"], observation["Foll"]
        self_pred_a = acceleration(ego_vehicle=ego, front_vehicle=new_preceding, mode=mode)

        # The deceleration of the new following vehicle after the the LC should not be too big (negative)
        if new_following_pred_a < -CAV_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False, gain

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = acceleration(ego_vehicle=ego, front_vehicle=old_preceding, mode=mode)
        old_following_a = acceleration(ego_vehicle=old_following, front_vehicle=ego, mode=mode)
        old_following_pred_a = acceleration(ego_vehicle=old_following, front_vehicle=old_preceding, mode=mode)
        gain = self_pred_a - self_a + CAV_POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
        if gain <= CAV_LANE_CHANGE_MIN_ACC_GAIN:
            gain = None
            return False, gain

        # All clear, let's go!
        return True, gain

    # @profile
    def step(self):
        """Controller decide the next action for the vehicle.
        """        
        super().step()
        self.action, self.mode = self.decision(self.vehicle.observation.information)
        

# class SurrogateIDMAVController(IDMAVController):
#     lanechange_list = {
#         0: "left",
#         1: "right"
#     }
#     @staticmethod
#     def CAV_decision(observation):
#         """CAV decides next action based on surrogate IDM model.

#         Args:
#             observation (dict): Observation of CAV.

#         Returns:
#             float: Action index. 0 represents left turn and 1 represents right turn. If the output is larger than 2, it is equal to the longitudinal acceleration plus 6.
#         """        
#         action = None
#         mode = None
#         ego_vehicle = observation["Ego"]
#         front_vehicle = observation["Lead"]   
#         CAV_left_prob, CAV_still_prob, CAV_right_prob = NADEBVGlobalController._get_Surrogate_CAV_action_probability(observation)    
#         SM_LC_Prob = [CAV_left_prob, CAV_still_prob, CAV_right_prob]
#         final_decision = np.random.choice([0, 2, 1], p=SM_LC_Prob).item()
        
#         # Lateral: MOBIL
#         if final_decision == 0 or final_decision == 1:
#             mode = "CAV_MOBIL"
#             action = {
#                 "lateral": self.lanechange_list[final_decision],
#                 "longitudinal": 0
#             }
#         # Longitudinal: IDM
#         else:
#             mode = "CAV_IDM"
#             tmp_acc = acceleration(ego_vehicle=ego_vehicle,front_vehicle=front_vehicle,mode="CAV")
#             tmp_acc = np.clip(tmp_acc, acc_low, acc_high)
#             action = {
#                 "lateral": "central",
#                 "longitudinal": tmp_acc
#             }
#         return action, mode

# class ACMIDMController(IDMAVController):

#     def step(self):
#         super().step()
#         if self.vehicle.simulator.getRoadID(self.vehicle.id) == "-1003000.207.30":
#             self.vehicle.simulator.changeTarget(self.vehicle.id, "-1008000.289.23")
#         else:
#             self.vehicle.simulator.changeTarget(self.vehicle.id, "-1003000.207.30")