import uuid
import numpy as np
import math
import bisect
from conf import conf
import time, os

def get_conf():
    episode_num = int(conf.experiment_config["episode_num"])
    root_folder = conf.experiment_config["root_folder"]
    experiment_name = conf.experiment_config["experiment_name"]
    full_experiment_name = "Experiment-%s" % experiment_name + \
        "_"+time.strftime('%Y-%m-%d', time.localtime(time.time()))
    experiment_path = os.path.join(root_folder, full_experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    conf_path = os.path.join(experiment_path, "conf")
    crash_path = os.path.join(experiment_path, "crash")
    rejected_path = os.path.join(experiment_path, "rejected")
    tested_but_safe_path = os.path.join(experiment_path, "tested_and_safe")
    for path_tmp in [conf_path, crash_path, rejected_path, tested_but_safe_path]:
        os.makedirs(path_tmp, exist_ok=True)
    return episode_num, experiment_path
    
    


def is_lane_change(obs_ego):
    if "prev_action" not in obs_ego:
        return False
    return (obs_ego["prev_action"] is not None and obs_ego["prev_action"]["lateral"] != "central")


def _Mobil_surraget_model(cav_obs, lane_index):
    """Apply the Mobil surrogate model for CAV Lane change to calculate the gain for this lane change maneuver. If it does not have safety issue, then return True, gain; otherwise False, None.

    Args:
        lane_index (integer): Candidate lane for the change.

    Returns:
        (bool, float): The first output stands for safety flag (whether ADS will crash immediately after doing LC), the second output is gain (Now could even smaller than 0).
    """
    gain = None
    cav_info = cav_obs['Ego']

    # Is the maneuver unsafe for the new following vehicle?
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

    new_following_a = acceleration(
        ego_vehicle=new_following, front_vehicle=new_preceding)
    new_following_pred_a = acceleration(
        ego_vehicle=new_following, front_vehicle=cav_info)

    old_preceding = cav_obs["Lead"]
    old_following = cav_obs["Foll"]
    self_pred_a = acceleration(
        ego_vehicle=cav_info, front_vehicle=new_preceding)

    # The deceleration of the new following vehicle after the the LC should not be too big (negative)
    if new_following_pred_a < -conf.Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED:
        return True, 0

    # Is there an acceleration advantage for me and/or my followers to change lane?
    self_a = acceleration(ego_vehicle=cav_info, front_vehicle=old_preceding)
    old_following_a = acceleration(
        ego_vehicle=old_following, front_vehicle=cav_info)
    old_following_pred_a = acceleration(
        ego_vehicle=old_following, front_vehicle=old_preceding)
    gain = self_pred_a - self_a + conf.Surrogate_POLITENESS * \
        (new_following_pred_a - new_following_a +
         old_following_pred_a - old_following_a)
    return True, gain


def _get_Surrogate_CAV_action_probability(cav_obs):
    """Obtain the lane change probability of CAV. If ADS will not immediately crash, then the LC probability is at least epsilon_lane_change_prob map gain from [0, 1] to LC probability [epsilon_lane_change_prob, max_remaining_LC_prob].

    Returns:
        (float, float, float): left-lane-change possibility, stay-still possibility, right-lane-change possibility.
    """
    CAV_left_prob, CAV_right_prob = 0, 0
    CAV_still_prob = conf.epsilon_still_prob
    left_gain, right_gain = 0, 0
    left_LC_safety_flag, right_LC_safety_flag = False, False
    # CAV will do lane change or not?
    lane_index_list = [-1, 1]  # -1: right turn; 1: left turn
    for lane_index in lane_index_list:
        LC_safety_flag, gain = _Mobil_surraget_model(
            cav_obs=cav_obs, lane_index=lane_index)
        # left_LC_safety_flag, gain = IDMAVController.mobil_gain(lane_index=lane_index, cav_obs = cav_obs)
        if gain is not None:
            if lane_index == -1:
                right_gain = np.clip(gain, 0., None)
                right_LC_safety_flag = LC_safety_flag
            elif lane_index == 1:
                left_gain = np.clip(gain, 0., None)
                left_LC_safety_flag = LC_safety_flag
    assert(left_gain >= 0 and right_gain >= 0)

    # ! quick fix the CAV lane change at one side result
    if not cav_obs["Ego"]["could_drive_adjacent_lane_left"]:
        left_LC_safety_flag = 0
        left_gain = 0
    elif not cav_obs["Ego"]["could_drive_adjacent_lane_right"] == 0:
        right_LC_safety_flag = 0
        right_gain = 0

    # epsilon LC probability if no safety issue and feasible for LC
    CAV_left_prob += conf.epsilon_lane_change_prob*left_LC_safety_flag
    CAV_right_prob += conf.epsilon_lane_change_prob*right_LC_safety_flag

    max_remaining_LC_prob = 1-conf.epsilon_still_prob-CAV_left_prob-CAV_right_prob

    total_gain = left_gain+right_gain
    obtained_LC_prob_for_sharing = np.clip(remap(total_gain, [0, conf.SM_MOBIL_max_gain_threshold], [
                                           0, max_remaining_LC_prob]), 0, max_remaining_LC_prob)
    CAV_still_prob += (max_remaining_LC_prob - obtained_LC_prob_for_sharing)

    if total_gain > 0:
        CAV_left_prob += obtained_LC_prob_for_sharing * \
            (left_gain/(left_gain + right_gain))
        CAV_right_prob += obtained_LC_prob_for_sharing * \
            (right_gain/(left_gain + right_gain))

    assert(0.99999 <= (CAV_left_prob + CAV_still_prob + CAV_right_prob) <= 1.0001)

    return CAV_left_prob, CAV_still_prob, CAV_right_prob


def action_id_to_action_command(action_id):
    action = None
    if action_id == 0:
        action = {
            "lateral": "left",
            "longitudinal": 0
        }
    elif action_id == 1:
        action = {
            "lateral": "right",
            "longitudinal": 0
        }
    else:
        action = {
            "lateral": "central",
            "longitudinal": float(conf.BV_ACTIONS[action_id])
        }
    return action


def generate_unique_bv_id():
    """Randomly generate an ID of the background vehicle

    Returns:
        string: ID of the background vehicle
    """
    return 'BV_'+str(uuid.uuid4())


def remap(v, x, y):
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])


def round_value_function(real_value, round_item):
    """Round a specific parameter based on the corresponding rule.

    Args:
        real_value (float): Real value.
        round_item (str): Parameter type.

    Returns:
        float/int: Rounded value.
    """
    if round_item == "speed":
        value_list = conf.speed_list
        value_dic = conf.v_to_idx_dic
    elif round_item == "range":
        value_list = conf.r_list
        value_dic = conf.r_to_idx_dic
    elif round_item == "range_rate":
        value_list = conf.rr_list
        value_dic = conf.rr_to_idx_dic

    if real_value < value_list[0]:
        real_value = value_list[0]
    elif real_value > value_list[-1]:
        real_value = value_list[-1]

    if round_item == "speed":
        value_idx = bisect.bisect_left(value_list, real_value)
        value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
        try:
            assert value_idx <= (len(value_list)-1)
            assert value_idx >= 0
        except:
            pass
        round_value = value_list[value_idx]
        assert value_dic[round_value] == value_idx
        return round_value, value_idx
    else:
        value_idx = bisect.bisect_left(value_list, real_value)
        value_idx = value_idx - \
            1 if real_value != value_list[value_idx] else value_idx
        try:
            assert value_idx <= (len(value_list)-1)
            assert value_idx >= 0
        except:
            pass
        round_value = value_list[value_idx]
        assert value_dic[round_value] == value_idx
        return round_value, value_idx


def _round_data(v, r, rr):
    """Round speed, range, and rangerate to the closest value based on resolution.

    Args:
        v (float): Speed [m/s].
        r (float): Vehicle range [m].
        rr (float): Range rate [m/s].

    Returns:
        float, int, int: Rounded speed, range, and range rate.
    """
    round_speed, _ = round_value_function(v, round_item="speed")
    round_r, _ = round_value_function(r, round_item="range")
    round_rr, _ = round_value_function(rr, round_item="range_rate")

    return round_speed, round_r, round_rr

def _round_data_plain(v, r, rr):
    return round(v), round(r), round(rr)

def _round_data_Haojie(v, r, rr):
    """Round the speed, vehicle range and range rate to the closest integer.

    Args:
        v (double): Speed [m/s].
        r (double): Vehicle range [m].
        rr (double): Range rate [m/s].

    Returns:
        (integer, integer, integer): Rounded speed [m/s], range [m] and range rate [m/s].
    """
    if v < conf.v_low:
        v_round = conf.v_low
    elif v > conf.v_high:
        v_round = conf.v_high
    else:
        v_round = int(v+0.5)
    if r < conf.r_low:
        r_round = conf.r_low
    elif r > conf.r_high:
        r_round = conf.r_high
    else:
        r_round = int(r+0.5)
    if rr < conf.rr_low:
        rr_round = conf.rr_low
    elif rr > conf.rr_high:
        rr_round = conf.rr_high
    else:
        if rr < 0:
            rr_round = int(rr-0.5)
        else:
            rr_round = int(rr+0.5)
    return v_round, r_round, rr_round


def epsilon_greedy(pdf_before_epsilon, ndd_pdf, epsilon=0.1):
    """defensive importance sampling: Owen, A. B. Monte Carlo Theory, Methods and Examples. https://statweb.stanford.edu/~owen/mc/  (2013).

    Args:
        pdf_before_epsilon (list): Importance function action probability distribution before epsilon greedy.
        ndd_pdf (list): Naturalistic action probability distribution.
        epsilon (float, optional): Epsilon value. Defaults to 0.1.

    Returns:
        list(float): Importance function action probability distribution after epsilon greedy.
    """
    pdf_after_epsilon = (1-epsilon) * pdf_before_epsilon + (epsilon) * ndd_pdf
    assert(0.99999 <= np.sum(pdf_after_epsilon) <= 1.0001)
    pdf_after_epsilon = pdf_after_epsilon/np.sum(pdf_after_epsilon)
    return pdf_after_epsilon


def check_equal(x, y, error):
    """Check if x is approximately equal to y considering the given error.

    Args:
        x (double): Parameter 1.
        y (double): Parameter 2.
        error (double): Specified error.

    Returns:
        bool: True is x and y are close enough. Otherwise, False.
    """
    if abs(x-y) <= error:
        return True
    else:
        return False


def cal_dis_with_start_end_speed(v_start, v_end, acc, time_interval=1.0):
    """Calculate the travel distance with start and end speed and acceleration.

    Args:
        v_start (float): Start speed [m/s].
        v_end (float): End speed [m/s].
        acc (float): Acceleration [m/s^2].
        time_interval (float, optional): Time interval [s]. Defaults to 1.0.

    Returns:
        float: Travel distance in the time interval.
    """
    if v_end == conf.v_low or v_end == conf.v_high:
        t_1 = (v_end-v_start)/acc if acc != 0 else 0
        t_2 = time_interval - t_1
        dis = v_start*t_1 + 0.5*(acc)*(t_1**2) + v_end*t_2
    else:
        dis = ((v_start+v_end)/2)*time_interval
    return dis


def pre_process_subscription(subscription, veh_id=None, distance=0.0):
    """Modify the subscription results into a standard form.

    Args:
        subscription (dict): Context subscription results of vehicle.
        veh_id (str, optional): Vehicle ID. Defaults to None.
        distance (float, optional): Distance from the ego vehicle [m]. Defaults to 0.0.

    Returns:
        dict: Standard for of vehicle information.
    """
    if not veh_id:
        return None
    veh = {"veh_id": veh_id}
    veh["position3D"] = subscription[veh_id][57]
    veh["velocity"] = subscription[veh_id][64]
    veh["position"] = subscription[veh_id][66]
    veh["heading"] = subscription[veh_id][67]
    veh["lane_index"] = subscription[veh_id][82]
    veh["distance"] = distance
    return veh


def combine_vehicle_information(veh_id, velocity, position, lane_index, distance=0.0):
    """Combine all the vehicle information to form a dictionary.

    Args:
        veh_id (str): Vehicle ID.
        velocity (float): Vehicle speed [m/s].
        position (tuple(float)): Vehicle position (x,y) [m].
        lane_index (int): Vehicle lane index.
        distance (float, optional): Distance between two vehicles [m]. Defaults to 0.0.

    Returns:
        dict: All vehicle information.
    """
    veh = {"veh_id": veh_id}
    veh["velocity"] = velocity
    veh["position"] = position
    veh["lane_index"] = lane_index
    veh["distance"] = distance
    return veh


def acceleration(ego_vehicle=None, front_vehicle=None, mode=None):
    """Compute an acceleration command with the Intelligent Driver Model. The acceleration is chosen so as to:
        - reach a target velocity;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

    Args:
        ego_vehicle (dict, optional): Information of the vehicle whose desired acceleration is to be computed. It does not have to be an IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to reason about other vehicles behaviors even though they may not IDMs. Defaults to None.
        front_vehicle (dict, optional): Information of the vehicle preceding the ego-vehicle. Defaults to None.
        mode (string, optional): Difference IDM parameters for BV and CAV. Defaults to None.

    Returns:
        float: Acceleration command for the ego-vehicle in m/s^2.
    """
    if not ego_vehicle:
        return 0

    if mode == "CAV":
        COMFORT_ACC_MAX = conf.CAV_COMFORT_ACC_MAX
        DESIRED_VELOCITY = conf.CAV_DESIRED_VELOCITY
        DELTA = conf.CAV_DELTA
    else:
        COMFORT_ACC_MAX = conf.COMFORT_ACC_MAX
        DESIRED_VELOCITY = conf.SM_IDM_DESIRED_VELOCITY
        DELTA = conf.SM_IDM_DELTA
    LENGTH = conf.LENGTH
    acceleration = COMFORT_ACC_MAX * \
        (1 - np.power(ego_vehicle["velocity"] /
                      DESIRED_VELOCITY, DELTA))
    if front_vehicle is not None:
        r = front_vehicle["position"][0] - ego_vehicle["position"][0]
        d = max(1e-5, r - LENGTH)
        acceleration -= COMFORT_ACC_MAX * \
            np.power(desired_gap(ego_vehicle, front_vehicle, mode) / d, 2)
    return acceleration


def desired_gap(ego_vehicle, front_vehicle=None, mode=None):
    """Compute the desired distance between a vehicle and its leading vehicle.

    Args:
        ego_vehicle (dict): Information of the controlled vehicle.
        front_vehicle (dict, optional): Information of the leading vehicle. Defaults to None.
        mode (string, optional): Difference IDM parameters for BV and CAV. Defaults to None.

    Returns:
        float: Desired distance between the two vehicles in m.
    """
    if mode == "CAV":
        d0 = conf.CAV_DISTANCE_WANTED
        tau = conf.CAV_TIME_WANTED
        ab = -conf.CAV_COMFORT_ACC_MAX * conf.CAV_COMFORT_ACC_MIN
    else:
        d0 = conf.DISTANCE_WANTED
        tau = conf.TIME_WANTED
        ab = -conf.COMFORT_ACC_MAX * conf.COMFORT_ACC_MIN
    dv = ego_vehicle["velocity"] - front_vehicle["velocity"]
    d_star = d0 + max(0, ego_vehicle["velocity"] * tau +
                      ego_vehicle["velocity"] * dv / (2 * np.sqrt(ab)))
    return d_star


def cal_euclidean_dist(veh1_position=None, veh2_position=None):
    """Calculate Euclidean distance between two vehicles.

    Args:
        veh1_position (tuple, optional): Position of Vehicle 1 [m]. Defaults to None.
        veh2_position (tuple, optional): Position of Vehicle 2 [m]. Defaults to None.

    Raises:
        ValueError: If the position of fewer than two vehicles are provided, raise error.

    Returns:
        Float: Euclidean distance between two vehicles [m].
    """
    if veh1_position is None or veh2_position is None:
        raise ValueError("Fewer than two vehicles are provided!")
    veh1_x, veh1_y = veh1_position[0], veh1_position[1]
    veh2_x, veh2_y = veh2_position[0], veh2_position[1]
    return math.sqrt(pow(veh1_x-veh2_x, 2)+pow(veh1_y-veh2_y, 2))


# class DuelingNet(nn.Module):
#     # for CAV
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(DuelingNet, self).__init__()
#         # print("Using Dueling Network!")
#         self.exp_item = 0  # marks the num that the expPool has stored
#         self.hidden = nn.Linear(n_feature, n_hidden)  # hidden layer
#         self.hidden2 = nn.Linear(n_hidden, n_hidden)
#         self.adv = nn.Linear(n_hidden, n_output)
#         self.val = nn.Linear(n_hidden, 1)
#         # self.hidden3 = nn.Linear(n_hidden, n_hidden)
#         # self.hidden4 = nn.Linear(n_hidden, n_hidden)

#     def forward(self, x):
#         """Forward function in neural networks for AV agent.

#         Args:
#             x ([type]): [description]

#         Returns:
#             [type]: [description]
#         """
#         # x = F.relu(self.hidden(x))  # activation function for hidden layer
#         # x = F.relu(self.hidden2(x))
#         x = torch.tanh(self.hidden(x))  # activation function for hidden layer
#         x = torch.tanh(self.hidden2(x))
#         adv = self.adv(x)
#         val = self.val(x)
#         # x = torch.tanh(self.hidden3(x))
#         # x = torch.tanh(self.hidden4(x))
#         x = val + adv - adv.mean()
#         return x
