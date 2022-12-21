from copy import copy
import numpy as np
from mtlsp.controller.vehicle_controller.controller import Controller
from mtlsp.observation.observation import Observation
import conf.conf as conf

class Vehicle(object):
    color_red = (255,0,0)
    color_yellow = (255,255,0)
    color_blue = (0,0,255)
    color_green = (0,255,0)
    r_low, r_high, rr_low, rr_high, acc_low, acc_high = 0, 115, -10, 8, -4, 2

    def __init__(self, id, controller, observation_method=Observation, routeID=None, simulator=None, initial_speed=None, initial_position=None, initial_lane_id=None):
        if conf.simulation_config["speed_mode"] == "low_speed":
            self.v_low, self.v_high = 0, 20
        elif conf.simulation_config["speed_mode"] == "high_speed":
            self.v_low, self.v_high = 20, 40
        self.id = id
        self.controller = controller
        controller.vehicle = self
        self.observation_method = observation_method
        self._recent_observation = None
        self.simulator = simulator
        self.action_step_size = self.simulator.action_step_size
        self.step_size = self.simulator.step_size
        self.lc_duration = self.simulator.lc_duration
        self.lc_step_num = int(self.simulator.lc_duration / self.action_step_size)
        self.routeID = routeID
        self.initial_speed = initial_speed
        self.initial_position = initial_position
        self.initial_lane_id = initial_lane_id
        self.controlled_flag = False
        self.controlled_duration = 0
        self.target_lane_index = None

    def __iter__(self):
        yield self

    @property
    def type(self):
        """Separate BV with AV through the type property.
        Returns:
            str: Type of the vehicle based on its ID. "BV" stands for background vehicle, "AV" stands for autonomous vehicle, "Unknown" stands for unknown type.
        """
        return self.id.split("_")[0]

    # @profile
    def install_controller(self, controller):
        """Install controller for each vehicle and change vehicle mode based on controller type.

        Args:
            controller (Controller): Controller object for the vehicle.
        """        
        self.controller = controller
        controller.vehicle = self
        self.controller.install()

    def __str__(self):
        return f'Vehicle(id: {self.id})'

    def __repr__(self):
        return self.__str__()

    # @profile
    def act(self, action):
        """Vehicle acts based on the input action.

        Args:
            action (dict): Lonitudinal and lateral actions. It should have the format: {'longitudinal': float, 'lateral': str}. The longitudinal action is the longitudinal acceleration, which should be a float. The lateral action should be the lane change direction. 'central' represents no lane change. 'left' represents left lane change, and 'right' represents right lane change.
        """ 
        self.simulator.set_vehicle_speedmode(self.id, 0)
        self.simulator.set_vehicle_lanechangemode(self.id, 0)
        controlled_acc = action["longitudinal"]
        current_velocity = self.observation.information["Ego"]["velocity"]
        if current_velocity + controlled_acc > self.v_high:
            controlled_acc = self.v_high - current_velocity
        elif current_velocity + controlled_acc < self.v_low:
            controlled_acc = self.v_low - current_velocity
        
        
        if action["lateral"] == "central":
            current_lane_offset = self.simulator.get_vehicle_lateral_lane_position(self.id)
            self.simulator.change_vehicle_sublane_dist(self.id, -current_lane_offset, self.step_size)
            self.simulator.change_vehicle_speed(self.id, controlled_acc, self.action_step_size)
        else:
            self.simulator.change_vehicle_lane(self.id, action["lateral"], self.lc_duration)
            self.simulator.change_vehicle_speed(self.id, controlled_acc, self.lc_duration)
        
    def is_action_legal(self, action):
        if action["lateral"] == "left" and not self.simulator.get_vehicle_lane_adjacent(self.id, 1):
            return False
        if action["lateral"] == "right" and not self.simulator.get_vehicle_lane_adjacent(self.id, -1):
            return False
        return True

    # @profile
    def update(self):
        """Update the state of the background vehicle, including conducting actions, setting colors and maintain controlled_flag and controlled_duration.

        Args:
            action (int, optional): Action index for the controlled background vehicle, otherwise None. Defaults to None.
        """
        if self.controller.action is not None and not self.controlled_flag and self.is_action_legal(self.controller.action):
            # Control the vehicle for the first time.
            # print(self.controller.action)
            self.act(self.controller.action)
            # print(self.simulator.getLateralLanePosition(self.id))
            if self.controller.action["lateral"] == "left" or self.controller.action["lateral"] == "right":
                self.controlled_duration += 1
            self.controlled_flag = True
            
    def reset_control_state(self):
        """Reset the control state of the vehicle, including setting the controlled_flag to be false, and setting the vehicle color.
        """        
        if self.controlled_flag:
            if self.controlled_duration == 0:
                self.controller.action = None
                self.controlled_flag = False
                self.controller.reset()
            else:
                self.controlled_duration = (self.controlled_duration + 1)%self.lc_step_num
 
    @property
    def observation(self):
        """Observation of the vehicle.

        Returns:
            Observation: Information of the vehicle itself and its surroundings. 
        """        
        if not self._recent_observation or self._recent_observation.time_stamp != self.simulator.get_time(): #if the recent observation exists and the recent observation is not updated at the current timestamp, update the recent observation
            self._recent_observation = self._get_observation()
        return self._recent_observation

    # @profile
    def _get_observation(self):
        """Get observation of the vehicle at the last time step. Do not directly use this method, instead, use the observation() property method for efficient performance.

        Returns:
            Observation: Surrounding information the vehicle along with the information of itself, including vehicle ID, speed, position, lane index, and vehicle distance.
        """
        obs = self.observation_method(ego_id=self.id, time_stamp=self.simulator.get_time())
        obs.update(self.simulator.get_vehicle_context_subscription_results(self.id), self.simulator, self)
        return obs
    

class VehicleList(dict):
    def __init__(self, d):
        """A vehicle list that store vehicles. It derives from a dictionary so that one can call a certain vehicle in O(1) time. Rewrote the iter method so it can iterate as a list.
        """
        super().__init__(d)

    def __add__(self, another_vehicle_list):
        if not isinstance(another_vehicle_list, VehicleList):
            raise TypeError('VehicleList object can only be added to another VehicleList')
        vehicle_list = copy(self)
        keys = self.keys()
        for v in another_vehicle_list:
            if v.id in keys:
                print(f'WARNING: vehicle with same id {v.id} is added and overwrote the vehicle list')
            vehicle_list[v.id] = v
        return vehicle_list

    def add_vehicles(self, vlist):
        """Add vehicles to the vehicle list.

        Args:
            vlist (list(Vehicle)): List of Vehicle object or a single Vehicle object.
        """        
        for v in vlist:
            if v.id in self.keys():
                # print(f'WARNING: vehicle with same id {v.id} exists and this vehicle is dumped and not overriding the vehicle with same id in the original list')
                continue
            self[v.id] = v    

    def __iter__(self):
        for k in self.keys():
            yield self[k]