from abc import ABC
from mtlsp.simulator import Simulator

class Observation(ABC):
    """Observation class store the vehicle observations, the time_stamp object is essential to allow observation to only update once. 
    It is composed of the local information, context information, processed information and time stamp.
    local: a dictionary{ vehicle ID: subsribed results (dictionary)
    }
    context: a dictionary{ vehicle ID: subsribed results (dictionary)
    }
    information: a dictionary{
        'Ego': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': 0 [m]},
        'Lead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'Foll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'LeftLead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'RightLead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'LeftFoll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'RightFoll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
    }
    time_stamp is used to record simulation time and for lazy use.
    """
    def __init__(self, ego_id=None, time_stamp=-1):
        if not ego_id:
            raise ValueError("No ego vehicle ID is provided!")
        self.ego_id = ego_id
        self.local = None
        self.context = None
        self.information = None
        if time_stamp == -1:
            raise ValueError("Observation is used before simulation started!")
        self.time_stamp = time_stamp

    # @profile
    def update(self, subscription=None, simulator=None, vehicle=None):
        """Update the observation of vehicle.

        Args:
            subscription (dict, optional): Context information obtained from SUMO Traci. Defaults to None.

        Raises:
            ValueError: When supscription results are None, raise error.
        """        
        if not subscription:
            raise ValueError("No subscription results are provided!")
        #! Careful: self.local and self.context are just a shallow copy of the subscription results!!!
        self.local = {self.ego_id:subscription[self.ego_id]}
        self.context = {}
        for bv_id in subscription.keys():
            if bv_id != self.ego_id:
                self.context[bv_id] = subscription[bv_id]
        self.information = self.traci_based_process(subscription, simulator, vehicle)

    def traci_based_process(self, subscription=None, simulator=None, vehicle=None):
        if not subscription:
            raise ValueError("No subscription results are imported!")
        obs = {"Ego": Observation.pre_process_subscription(subscription, simulator, veh_id=self.ego_id, vehicle=vehicle)}
        obs["Lead"] = simulator.get_leading_vehicle(self.ego_id)
        obs["LeftLead"] = simulator.get_neighboring_leading_vehicle(self.ego_id, "left")
        obs["RightLead"] = simulator.get_neighboring_leading_vehicle(self.ego_id, "right")
        obs["Foll"] = simulator.get_following_vehicle(self.ego_id)
        obs["LeftFoll"] = simulator.get_neighboring_following_vehicle(self.ego_id, "left")
        obs["RightFoll"] = simulator.get_neighboring_following_vehicle(self.ego_id, "right")
        return obs

    @staticmethod
    # @profile
    def pre_process_subscription(subscription, simulator, veh_id=None, vehicle=None, distance=0.0):
        """Modify the subscription results into a standard form.

        Args:
            subscription (dict): Context subscription results of vehicle.
            simulator (Simulator): Simulator object.
            veh_id (str, optional): Vehicle ID. Defaults to None.
            distance (float, optional): Distance from the ego vehicle [m]. Defaults to 0.0.

        Returns:
            dict: Standard for of vehicle information.
        """
        if not veh_id:
            return None
        veh = {"veh_id": veh_id}

        veh["could_drive_adjacent_lane_left"] = simulator.get_vehicle_lane_adjacent(veh_id,1)
        veh["could_drive_adjacent_lane_right"] = simulator.get_vehicle_lane_adjacent(veh_id,-1)
        veh["distance"] = distance
        veh["heading"] = subscription[veh_id][67]
        veh["lane_index"] = subscription[veh_id][82]
        veh["lateral_speed"] = subscription[veh_id][50]
        veh["lateral_offset"] = subscription[veh_id][184]
        veh["prev_action"] = vehicle.controller.action
        veh["position"] = subscription[veh_id][66]
        veh["position3D"] = subscription[veh_id][57]
        veh["velocity"] = subscription[veh_id][64]
        veh["road_id"] = subscription[veh_id][80]
        veh["acceleration"] = subscription[veh_id][114]
        return veh