import uuid
import numpy as np
import math
import bisect


def generate_unique_bv_id():
    """Randomly generate an ID of the background vehicle

    Returns:
        str: ID of the background vehicle
    """
    return 'BV_'+str(uuid.uuid4())

def remap(v, x, y): 
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])

def check_equal(x, y, error):
    """Check if x is approximately equal to y considering the given error.

    Args:
        x (float): Parameter 1.
        y (float): Parameter 2.
        error (float): Specified error.

    Returns:
        bool: True is x and y are close enough. Otherwise, False.
    """
    if abs(x-y) <= error:
        return True
    else:
        return False

def cal_dis_with_start_end_speed(v_start, v_end, acc, time_interval=1.0, v_low=20, v_high=40):
    """Calculate the travel distance with start and end speed and acceleration.

    Args:
        v_start (float): Start speed [m/s].
        v_end (float): End speed [m/s].
        acc (float): Acceleration [m/s^2].
        time_interval (float, optional): Time interval [s]. Defaults to 1.0.

    Returns:
        float: Travel distance in the time interval.
    """
    if v_end == v_low or v_end == v_high:
        t_1 = (v_end-v_start)/acc if acc != 0 else 0
        t_2 = time_interval - t_1
        dis = v_start*t_1 + 0.5*(acc)*(t_1**2) + v_end*t_2
    else:
        dis = ((v_start+v_end)/2)*time_interval
    return dis

def cal_euclidean_dist(veh1_position=None, veh2_position=None):
    """Calculate Euclidean distance between two vehicles.

    Args:
        veh1_position (tuple, optional): Position of Vehicle 1 [m]. Defaults to None.
        veh2_position (tuple, optional): Position of Vehicle 2 [m]. Defaults to None.

    Raises:
        ValueError: If the position of fewer than two vehicles are provided, raise error.

    Returns:
        float: Euclidean distance between two vehicles [m].
    """    
    if veh1_position is None or veh2_position is None:
        raise ValueError("Fewer than two vehicles are provided!")
    veh1_x, veh1_y = veh1_position[0], veh1_position[1]
    veh2_x, veh2_y = veh2_position[0], veh2_position[1]
    return math.sqrt(pow(veh1_x-veh2_x, 2)+pow(veh1_y-veh2_y, 2))