# from tensorboardX import SummaryWriter
from bidict import bidict
import os
import numpy as np
import scipy.io
import pickle

print_log_flag = False
simulation_config = {}
simulation_config["multistep_flag"] = False
simulation_config["initialization_rejection_sampling_flag"] = False
# simulation_config["fast_intersection_check"] = False
simulation_config["gui_flag"] = False
simulation_config["step_size"] = 0.1

traffic_flow_config = {}
traffic_flow_config["CAV"] = True
traffic_flow_config["BV"] = True


treesearch_config = {}
treesearch_config["surrogate_model"] = "surrogate"
# treesearch_config["surrogate_model"] = "AVI"
treesearch_config["search_depth"] = 2
treesearch_config["offline_leaf_evaluation"] = False
treesearch_config["offline_discount_factor"] = 1
treesearch_config["treesearch_discount_factor"] = 1

experiment_config = {}
experiment_config["AV_model"] = "IDM" #"IDM", "Surrogate"
experiment_config["root_folder"] = "/home/mtl/DEV_HAOWEI/NADE/output"
experiment_config["experiment_name"] = "default_experiment_name"
experiment_config["MP_process_num"] = 4
experiment_config["log_mode"] = "crash" # only log crash 
experiment_config["episode_num"] = 200000
experiment_config["input_path"] = None
experiment_config["log_initial_criticality"] = False


# lane_list = np.array([42.0, 46.0, 50.0])
lane_list = np.array([42.0, 46.0])
# =========== Simulation parameters ==========
simulation_resolution = 1.0  # The time interval(resolution) of the simulation platform (second)
# =========== Vehicle parameters =============
LENGTH = 5.0
# vehicle color rgb code
color_red = (255,0,0)
color_yellow = (255,255,0)
color_blue = (0,0,255)
color_green = (0,255,0)
# =========== Highway parameters =============
HIGHWAY_LENGTH = 1200  # 1200 1600 4000
EXIT_LENGTH = 800  # 800 1400
v_min, v_max = 20, 40  # m/s vehicles speed limit
a_min, a_max = -4, 2

# =========== Initialization of env ===========
random_initialization_vehicles_count = 18 # ! Not used
random_initialization_BV_v_min, random_initialization_BV_v_max = 28, 32 # ! Not used
random_env_BV_generation_range_min, random_env_BV_generation_range_max = 0, 300  # ! Not used
initial_CAV_speed = 30   # ! Not used
initial_CAV_position = 400  # ! Not used
ndd_data_path = "./source_data"

CF_joint_pdf = np.load(ndd_data_path + "/NDD_DATA/CF/Optimized_CF_joint_pdf_array.npy")  # ! Not used
presum_list_forward = np.load(ndd_data_path + "/NDD_DATA/Initialization/Optimized_presum_list_forward.npy") # choose prev vehicle according to range and range rate, according to the velocity of the following vehicle
    
# =========== NDD information ===========
CF_percent = 0.6823141  # how many vehicles are in car following mode, otherwise in free flow mode
ff_dis = 120  #  free flow minimum distance
gen_length = 1800 # the traffic flow will not be generated after this length alongside the road from the begining point
acm_gen_length = 3200 # ! Not used
long_gen_length = 4800 # ! Not used
random_veh_pos_buffer_start, random_veh_pos_buffer_end = 0, 50 # the first vehicle in the traffic flow will be generated between random_veh_pos_buffer_start and random_veh_pos_buffer_end
bv_obs_range = 120 # ! Not used
cav_obs_range = 120 # cav will only observe vehicles in this range, also, only vehicles in this range will be selected as nade candidates (120m is sufficient for highway scenarios)
round_rule = "Round_to_closest" # how continious variables are rounded, currently the rule is "Round_to_closest"

speed_CDF = list(np.load(ndd_data_path + "/NDD_DATA/Initialization/speed_CDF.npy"))  # NDD distribution to generate the initial speed of ndd vehicles
CF_pdf_array = np.load(ndd_data_path + "/NDD_DATA/CF/Optimized_CF_pdf_array.npy") # NDD distribution to generate the longitudinal decision (acceleration) under car following scenarios
FF_pdf_array = np.load(ndd_data_path + "/NDD_DATA/FF/Optimized_FF_pdf_array.npy") # NDD distribution to generate longitudinal decisions (acceleration) under free flow scenarios

LC_smooth_flag = True # ! Not used
Sparse_flag = True # ! Not used

# =========== NDD distribution to generate lateral decisions (lane change) ===========
import sparse
OL_pdf = sparse.load_npz(ndd_data_path + "/NDD_DATA/LC/10106_10620_OL_pdf_smoothed_soft_p8_1_tmp.npz").todense()
SLC_pdf = sparse.load_npz(ndd_data_path + "/NDD_DATA/LC/10119_10630_SLC_pdf_smoothed_soft_p8_1_tmp.npz").todense()
DLC_pdf = sparse.load_npz(ndd_data_path + "/NDD_DATA/LC/10119_10630_DLC_pdf_smoothed_p8_1_tmp.npz").todense()
CI_pdf = sparse.load_npz(ndd_data_path + "/NDD_DATA/LC/10119_10630_CI_pdf_smoothed_p8_1_tmp.npz").todense()

print("================Load Action NDD data finished!=================")


# =========== NDD distribution for car following / free flow ===========
v_low, v_high, r_low, r_high, rr_low, rr_high, acc_low, acc_high = 20, 40, 0, 115, -10, 8, -4, 2
v_resolution, r_resolution, rr_resolution, acc_resolution = 1, 1, 1, 0.2
if print_log_flag:
    if v_resolution == 0.2: print("Action v resolution is 0.2!")
    if v_resolution == 1: print("Action v resolution is 1!")
num_v, num_r, num_rr, num_acc = int(1+((v_high-v_low)/v_resolution)), int(1+((r_high-r_low)/r_resolution)),\
                                int(1+((rr_high-rr_low)/rr_resolution)), int(1+((acc_high-acc_low)/acc_resolution))
assert(((num_r, num_rr, num_v, num_acc) == CF_pdf_array.shape) and ((num_v, num_acc) == FF_pdf_array.shape))
speed_list, r_list, rr_list, acc_list = list(np.linspace(v_low, v_high, num=num_v)), list(np.linspace(r_low, r_high, num=num_r)),\
                                    list(np.linspace(rr_low, rr_high, num=num_rr)), list(np.linspace(acc_low, acc_high, num=num_acc))
r_to_idx_dic, rr_to_idx_dic, v_to_idx_dic, v_back_to_idx_dic, acc_to_idx_dic = bidict(), bidict(), bidict(), bidict(), bidict()  # Two way dictionary, the front path is: key: real value, value: idx
for i in range(num_r): r_to_idx_dic[list(np.linspace(r_low, r_high, num=num_r))[i]] = i
for j in range(num_rr): rr_to_idx_dic[list(np.linspace(rr_low, rr_high, num=num_rr))[j]] = j
for k in range(num_v): v_to_idx_dic[list(np.linspace(v_low, v_high, num=num_v))[k]] = k
for m in range(num_acc): acc_to_idx_dic[list(np.linspace(acc_low, acc_high, num=num_acc))[m]] = m

# NDD distribution for One lead lateral decision
one_lead_v_to_idx_dic, one_lead_r_to_idx_dic, one_lead_rr_to_idx_dic = bidict(), bidict(), bidict()
one_lead_r_low, one_lead_r_high, one_lead_rr_low, one_lead_rr_high, one_lead_v_low, one_lead_v_high = 0, 115, -10, 8, 20, 40
one_lead_r_step, one_lead_rr_step, one_lead_v_step = 1, 1, 1
one_lead_speed_list, one_lead_r_list, one_lead_rr_list = list(range(one_lead_v_low, one_lead_v_high+one_lead_v_step, one_lead_v_step)), list(range(one_lead_r_low, one_lead_r_high+one_lead_r_step, one_lead_r_step)), list(range(one_lead_rr_low, one_lead_rr_high+one_lead_rr_step,one_lead_rr_step))
for i in range(int((one_lead_v_high-one_lead_v_low+one_lead_v_step)/one_lead_v_step)): one_lead_v_to_idx_dic[list(range(one_lead_v_low, one_lead_v_high+ one_lead_v_step, one_lead_v_step))[i]] = i
for i in range(int((one_lead_r_high-one_lead_r_low+one_lead_r_step)/one_lead_r_step)): one_lead_r_to_idx_dic[list(range(one_lead_r_low, one_lead_r_high+one_lead_r_step, one_lead_r_step))[i]] = i
for i in range(int((one_lead_rr_high-one_lead_rr_low+one_lead_rr_step)/one_lead_rr_step)): one_lead_rr_to_idx_dic[list(range(one_lead_rr_low, one_lead_rr_high+one_lead_rr_step, one_lead_rr_step))[i]] = i

# NDD distribution for Double lane change lateral decision
lc_v_low, lc_v_high, lc_v_num, lc_v_to_idx_dic = 20, 40, 21, bidict()
lc_v_list = list(np.linspace(lc_v_low,lc_v_high,num=lc_v_num)) 
lc_rf_low, lc_rf_high, lc_rf_num, lc_rf_to_idx_dic = 0, 115, 116, bidict()
lc_rf_list = list(np.linspace(lc_rf_low,lc_rf_high,num=lc_rf_num))
lc_rrf_low, lc_rrf_high, lc_rrf_num, lc_rrf_to_idx_dic = -10, 8, 19, bidict()
lc_rrf_list = list(np.linspace(lc_rrf_low,lc_rrf_high,num=lc_rrf_num))
lc_re_low, lc_re_high, lc_re_num, lc_re_to_idx_dic = 0, 115, 116, bidict()
lc_re_list = list(np.linspace(lc_re_low,lc_re_high,num=lc_re_num))
lc_rre_low, lc_rre_high, lc_rre_num, lc_rre_to_idx_dic = -10, 8, 19, bidict()
lc_rre_list = list(np.linspace(lc_rre_low,lc_rre_high,num=lc_rre_num))
for i in range(lc_v_num): lc_v_to_idx_dic[list(np.linspace(lc_v_low,lc_v_high,num=lc_v_num))[i]] = i
for i in range(lc_rf_num): lc_rf_to_idx_dic[list(np.linspace(lc_rf_low,lc_rf_high,num=lc_rf_num))[i]] = i
for i in range(lc_re_num): lc_re_to_idx_dic[list(np.linspace(lc_re_low,lc_re_high,num=lc_re_num))[i]] = i
for i in range(lc_rrf_num):lc_rrf_to_idx_dic[list(np.linspace(lc_rrf_low,lc_rrf_high,num=lc_rrf_num))[i]] = i
for i in range(lc_rre_num):lc_rre_to_idx_dic[list(np.linspace(lc_rre_low,lc_rre_high,num=lc_rre_num))[i]] = i


# =========== Other parameters ============
Network_type = "Dueling"  # ! Not used
train_num = 0 # ! Not used
effective_test_item = 0 # ! Not used
total_train_num_for_epsilon = 2e5  # ! Not used
Min_Mem_For_Train = 5000 # ! Not used
validation_num = 500 # ! Not used
validation_freq = 10000 # ! Not used
save_memory_freq = 30000 # ! Not used
min_alpha, max_alpha, fully_compensate_train_num = 0.5, 1, 1e6 # ! Not used
# =========== Restore training parameters =======
cav_restore_flag = False # ! Not used
bv_restore_flag = False # ! Not used
cav_restore_training_num = 0 # ! Not used
bv_restore_training_num = 0 # ! Not used
bv_restore_file_folder = "BV_DQN_RESULT_FIX_INI_ONE_VEH" # ! Not used
change_lr_flag = False # ! Not used
# =========== Accelerate training parameters =======
Acc_training_flag = False # ! Not used

# =========== Save address ==============

save_folder_name = "CAV_Dueling_SG_1400_1600_1e-5_256_64" # ! Not used
bv_save_folder_name = "BV_DQN_RESULT_FIX_INI_SOFTMAX_Q" # ! Not used
cav_training = False # ! Not used
bv_training = False # ! Not used

# ! Not used
if cav_training:
    # Check existence of the save file address
    main_folder = os.path.abspath(".") + "/models/CAV/" + save_folder_name
    save_folder_dev = os.path.abspath(".") + "/models/CAV/" + save_folder_name + "/dev/"
    save_folder_target_net = os.path.abspath(".") + "/models/CAV/" + save_folder_name + "/target_net/"
    save_folder_memory = os.path.abspath(".") + "/models/CAV/" + save_folder_name + "/memory/"
    if os.path.exists(main_folder):pass
    else: os.mkdir(main_folder)
    if os.path.exists(save_folder_dev): pass
    else: os.mkdir(save_folder_dev)
    if os.path.exists(save_folder_target_net): pass
    else: os.mkdir(save_folder_target_net)
    if os.path.exists(save_folder_memory): pass
    else: os.mkdir(save_folder_memory)
# ! Not used
if bv_training:
    bv_main_folder = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name
    bv_save_folder_dev = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name + "/dev/"
    bv_save_folder_target_net = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name + "/target_net/"
    bv_save_folder_memory = os.path.abspath(".") + "/models/BV/" + bv_save_folder_name + "/memory/"
    if os.path.exists(bv_main_folder):pass
    else: os.mkdir(bv_main_folder)
    if os.path.exists(bv_save_folder_dev): pass
    else: os.mkdir(bv_save_folder_dev)
    if os.path.exists(bv_save_folder_target_net): pass
    else: os.mkdir(bv_save_folder_target_net)
    if os.path.exists(bv_save_folder_memory): pass
    else: os.mkdir(bv_save_folder_memory)

# =========== BV related ================
BV_ACTIONS = {0: 'LANE_LEFT',
            1: 'LANE_RIGHT'} # lateral and longitudinal maneuver candidates of BV
num_acc = int(((acc_high - acc_low)/acc_resolution) + 1) # number of longitudinal maneuver candidates (31 discrete accelerations) of BV
num_non_acc = len(BV_ACTIONS) # number of lateral maneuver candidates (left lane change and right lane change) of BV
for i in range(num_acc):
    acc = acc_to_idx_dic.inverse[i]
    BV_ACTIONS[i+num_non_acc] = str(acc)
BV_ACTION_STEP_NUM = 10 # ! Not used
BV_ACTION_STEP = BV_ACTION_STEP_NUM*simulation_config["step_size"] # ! Not used
# =========== CAV related ================
# ACTIONS = {0: 'LANE_LEFT',
#                1: 'LANE_RIGHT',
#                2: '-4',
#                3: '-3',
#                4: '-2',
#                5: '-1',
#                6: '0',
#                7: '1',
#                8: '2'}
ACTIONS = {0: 'LANE_LEFT',
            1: 'LANE_RIGHT'} # lateral and longitudinal maneuver candidates of CAV
num_acc = int(((acc_high - acc_low)/acc_resolution) + 1) # number of longitudinal maneuver candidates (31 discrete accelerations) of AV
num_non_acc = len(ACTIONS) # number of lateral maneuver candidates (left lane change and right lane change) of AV
for i in range(num_acc):
    acc = acc_to_idx_dic.inverse[i]
    ACTIONS[i+num_non_acc] = str(acc)

# Longitudinal CAV policy parameters
CAV_COMFORT_ACC_MAX = 1.5 # maximum comfortable acceleration for CAV IDM model
CAV_COMFORT_ACC_MIN = -2 # minimum comfortable acceleration for CAV IDM model
CAV_DISTANCE_WANTED = 2.0 #5.0  # [m]
CAV_TIME_WANTED = 1.2 #1.5  # [s]
CAV_DESIRED_VELOCITY = 33.33 #35 # [m/s]
CAV_DELTA = 4.0  # []

# Lateral CAV policy parameters (MOBIL)
CAV_POLITENESS = 0.  # in [0, 1]
CAV_LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
CAV_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
CAV_LANE_CHANGE_DELAY = 1.0  # [s]

CAV_acc_low, CAV_acc_high, CAV_acc_step = -4, 2, 0.2 # ! Not used
num_CAV_acc = int((CAV_acc_high - CAV_acc_low)/CAV_acc_step + 1) # ! Not used
CAV_acc_to_idx_dic = bidict() # ! Not used
for i in range(num_CAV_acc): CAV_acc_to_idx_dic[list(np.arange(CAV_acc_low, CAV_acc_high + CAV_acc_step, CAV_acc_step))[i]] = i # ! Not used
cav_observation_num = 10 # ! Not used
ACTION_STEP = 1.0 # ! Not used

# =========== NDD ENV para ============
safety_guard_enabled_flag, safety_guard_enabled_flag_IDM = False, True # In car following scenarios, if the current state is out of the NDD data bound, or current state has no NDD data, we will use the stochastic IDM model to generate the longitudinal decision, and the safety guard function will be utilized to ensure the safety of the longitudinal decision.
Initial_range_adjustment_SG = 0  # ! Not used
Initial_range_adjustment_AT = 0  # ! Not used       
Stochastic_IDM_threshold = 1e-10 # In the longitudinal distribution generated by the stochastic IDM model, only maneuvers with probability larger than this threshold (1e-10) will be considered as available maneuver candidates, this will filter some extremely small probabilities (e.g., 1e-20)

# NDD Vehicle IDM parameters
COMFORT_ACC_MAX = 2 # [m/s2]
COMFORT_ACC_MIN = -4.0  # [m/s2]
DISTANCE_WANTED = 5.0  # [m]
TIME_WANTED = 1.5  # [s]
DESIRED_VELOCITY = 35 # [m/s]
DELTA = 4.0  # []

# NDD Vehicle MOBIL parameters
NDD_POLITENESS = 0.1  # in [0, 1]  0.5
NDD_LANE_CHANGE_MIN_ACC_GAIN = 0.2 # [m/s2]  0.2
NDD_LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0  # [m/s2]  2

# =========== Test parameters ==============
r_threshold_NDD = 0  # ! Not used
longi_safety_buffer, lateral_safety_buffer = 2, 2  # The safety buffer used to longitudinal and lateral safety guard

dis_with_CAV_in_critical_initialization = 300 # ! Not used
critical_ini_start = initial_CAV_position - dis_with_CAV_in_critical_initialization # ! Not used
critical_ini_end = initial_CAV_position  # ! Not used

LANE_CHANGE_SCALING_FACTOR = 1. # ! Not used
# =========== whether to enable different lane change modes ============
enable_One_lead_LC = True
enable_Single_LC = True
enable_Double_LC = True
enable_Cut_in_LC = True
enable_MOBIL = True
double_LC_prob_in_leftmost_rightest_flag = False # In the leftest or rightest, double the other lane change probability, currently this feature is disabled
LC_num_in_simulation = 0  # ! Not used

Cut_in_veh_adj_rear_threshold = 5  # ! Not used
ignore_adj_veh_prob, min_r_ignore = 1e-2, 5 # Probability of the ignoring the vehicle in the adjacent lane
ignore_lane_conflict_prob = 1e-4  # ! Not used
LC_range_threshold = 200 # ! Not used

# ============= Criticality Model Data ================
CF_state_value = scipy.io.loadmat(ndd_data_path + "/Decompose/CF/" + "dangerous_state_table.mat")["dangerous_state_table"]
CF_challenge_value = scipy.io.loadmat(ndd_data_path + "/Decompose/CF/" + "Q_table_little.mat")["Q_table_little"]
# CF_state_value = scipy.io.loadmat(ndd_data_path + "/Decompose/CF/" + "dangerous_state_table_1e-1_gap.mat")["dangerous_state_table"]
# CF_challenge_value = scipy.io.loadmat(ndd_data_path + "/Decompose/CF/" + "Q_table_little_1e-1_gap.mat")["Q_table_little"]

BV_CF_state_value = np.load(ndd_data_path + "/Decompose/BV_CF/BV_CF_state_value.npy")
BV_CF_challenge_value = np.load(ndd_data_path + "/Decompose/BV_CF/BV_CF_challenge_value.npy")

episode = 0 # ! Not used

# CAV surrogate model IDM parameter
SM_IDM_COMFORT_ACC_MAX = 2.0  # ! Not used
SM_IDM_COMFORT_ACC_MIN = -4.0  # ! Not used
SM_IDM_DISTANCE_WANTED = 5.0  # ! Not used
SM_IDM_TIME_WANTED = 1.5  # ! Not used
SM_IDM_DESIRED_VELOCITY = 35 # [m/s]
SM_IDM_DELTA = 4.0  # []
epsilon_still_prob = 0.1 # 10% keep still even if Mobil model sets to do lane change
epsilon_lane_change_prob = 1e-8  # 5% to do lane change even if the gain is zero or even negative
SM_MOBIL_max_gain_threshold = 1 # m/s^2 when gain is greater than this, the LC probability will be maximum
Surrogate_POLITENESS = 0.
criticality_threshold = 0
# epsilon_value = 0.1
weight_threshold = 1e-8

OL_LC_low_speed_flag, OL_LC_low_speed_use_v = False, 24 # ! Not used



