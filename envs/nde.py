import os
import bisect
import numpy as np
from mtlsp.envs.env import BaseEnv
from mtlsp.controller.vehicle_controller.controller import Controller
from mtlsp.controller.vehicle_controller.idmcontroller import IDMController
from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from mtlsp.vehicle.vehicle import Vehicle
from mtlsp.logger.infoextractor import InfoExtractor
from mtlsp import utils
import conf.conf as conf
from controller.nddcontroller import NDDController
from controller.nddglobalcontroller import NDDBVGlobalController
import copy

class NDE(BaseEnv):
    def __init__(self, AVController=IDMController, BVController=NDDController, AVGlobalController=DummyGlobalController, BVGlobalController=NDDBVGlobalController, info_extractor=InfoExtractor):
        self.default_av_controller = AVController
        super().__init__(
            global_controller_dict={
                "BV": BVGlobalController, "CAV": AVGlobalController},
            independent_controller_dict={
                "BV": BVController, "CAV": AVController},
            info_extractor=info_extractor
        )

    def initialize(self):
        """Initialize the NDE simulation.
        """
        super().initialize()
        self.soft_reboot()
        for veh_id in self.simulator.get_vehID_list():
            self.simulator.set_vehicle_max_lateralspeed(veh_id, 4.0)

    # @profile
    def _step(self):
        """NDE simulation step.
        """
        return super()._step()
    
    def get_av_obs(self):
        cav = self.vehicle_list["CAV"]
        return_information = copy.deepcopy(cav.observation.information)
        try:
            for veh in return_information:
                try:
                    if return_information[veh] and return_information[veh]["veh_id"] != "CAV": # has vehicle
                        veh_id = return_information[veh]["veh_id"]
                        ndd_pdf = np.array(self.vehicle_list[veh_id].controller.ndd_pdf)
                        return_information[veh]["ndd_pdf"] = ndd_pdf.tolist()
                except Exception as e:
                    print(e)
        except:
            pass
        return return_information

    def get_av_ttc(self):
        cav = self.vehicle_list["CAV"]
        cav_observation = cav.observation.information
        if cav_observation["Lead"] is not None:
            distance_front, ttc_front = self.get_ttc(cav_observation["Lead"], cav_observation["Ego"])
        else:
            distance_front, ttc_front = 10000, 10000
        if cav_observation["Foll"] is not None:
            distance_back, ttc_back = self.get_ttc(cav_observation["Ego"], cav_observation["Foll"])
        else:
            distance_back, ttc_back = 10000, 10000
        return min(distance_front, distance_back), min(ttc_front, ttc_back)

    def get_ttc(self, lead_obs, follow_obs):
        distance = lead_obs["position"][0] - follow_obs["position"][0] - 5
        ttc = distance / (follow_obs["velocity"] - lead_obs["velocity"])
        if ttc < 0:
            return distance, 10000
        elif ttc > 10000:
            return distance, 10000
        else:
            return distance, ttc

    def add_background_vehicles(self, vlist, add_to_vlist=True, add_to_sumo=True):
        if add_to_vlist:
            self.vehicle_list.add_vehicles(vlist)
        if add_to_sumo:
            for v in vlist:
                self.simulator._add_vehicle_to_sumo(v, typeID='IDM')

    def soft_reboot(self):
        """Delete all vehicles and re-generate all vehicles on the road
        """
        self.generate_traffic_flow()
        # Focus on the CAV in the SUMO-GUI
        if self.simulator.track_cav:
            self.simulator.track_vehicle_gui()
            self.simulator.set_zoom(500)

    def generate_traffic_flow(self, init_info=None):
        """Generate traffic flow including one AV abd several BVs based on NDD.
        """
        if conf.traffic_flow_config["CAV"]:
            avID, avINFO = self.generate_av(
            controller_type=self.default_av_controller, speed=35)
        if conf.traffic_flow_config["BV"]:
            self.generate_bv_traffic_flow()

    def generate_av(self, speed=15.0, id="CAV", route="route_0", type_id="IDM", position=400.0, av_lane_id=None, controller_type=IDMController):
        """Generate one av in the network.

        Args:
            speed (float, optional): Initial speed. Defaults to 0.0.
            id (string, optional): CAV ID. Defaults to "CAV".
            route (string, optional): Route ID. Defaults to "route_0".
            type_id (string, optional): Vehicle type ID. Defaults to "IDM".
            position (float, optional): Initial position of the vehicle. Defaults to 400.0.
            controller_type (class, optional): Controller type of the AV. Defaults to AVController.
        """
        if av_lane_id is None:
            av_lane_id = self.simulator.get_available_lanes()[1].getID()
        av = Vehicle(id=id, controller=self.default_av_controller(), routeID=route, simulator=self.simulator,
                     initial_speed=speed, initial_position=position, initial_lane_id=av_lane_id)
        self.simulator._add_vehicle_to_sumo(av, typeID=type_id)
        av.install_controller(controller_type())
        self.vehicle_list.add_vehicles([av])
        info = {
            'speed': speed,
            'lane_id': av_lane_id,
            'route_id': route,
            'position': position
        }
        return id, info

    def generate_bv_traffic_flow(self):
        for lane in self.simulator.get_available_lanes():
            self.generate_ndd_flow_on_single_lane(lane_id=lane.getID())

    def generate_ndd_flow_on_single_lane(self, lane_id, gen_length=conf.gen_length):
        """Generate NDD vehicle flow one one single lane.

        Args:
            lane_id (str): Lane ID.
        """
        bv_before_av_generation_flag = False
        if ("CAV" not in self.vehicle_list) or self.vehicle_list["CAV"].initial_lane_id != lane_id: # if CAV is not on current lane
            tmp_speed_position = self.generate_ndd_vehicle(
                back_speed_position=None, lane_id=lane_id)
        else:
            cav_speed = self.vehicle_list["CAV"].initial_speed
            tmp_speed_position = {
                "speed": cav_speed, "position": self.vehicle_list["CAV"].initial_position}
            bv_before_av_generation_flag = True # will be disabled after the bv ahead of av is generated

        tmp_speed, tmp_position = tmp_speed_position["speed"], tmp_speed_position["position"]

        while tmp_position <= gen_length:
            tmp_speed_position = self.generate_ndd_vehicle(back_speed_position=tmp_speed_position, lane_id=lane_id, bv_before_av_generation_flag=bv_before_av_generation_flag)
            tmp_speed, tmp_position = tmp_speed_position["speed"], tmp_speed_position["position"]
            bv_before_av_generation_flag = False

    def generate_ndd_vehicle(self, back_speed_position=None, lane_id=None, bv_before_av_generation_flag=False):
        """This function will generate an vehicle under NDD distribution.

        Args:
            back_vehicle ([type], optional): [description]. Defaults to None.
        """
        if back_speed_position == None: # no vehicle behind
            speed, position = self.generate_random_vehicle()
        else:
            mode = self.sample_CF_FF_mode()
            if mode == "FF":
                speed_position = self.generate_FF_vehicle(back_speed_position)
                speed, position = speed_position["speed"], speed_position["position"]
            elif mode == "CF":
                speed_position = self.generate_CF_vehicle(back_speed_position)
                speed, position = speed_position["speed"], speed_position["position"]
            else:
                raise ValueError(
                    f"The vehicle mode needs to be CF/FF, however {mode} detected")

        if bv_before_av_generation_flag:
            print("generating vehicles")
            cav_speed_position = back_speed_position
            speed, position = self.safe_av_bv_generation(cav_speed_position["speed"], cav_speed_position["position"], speed, position)

        vehID = utils.generate_unique_bv_id()
        route = 'route_0'
        self.add_background_vehicles(Vehicle(vehID, controller=Controller(), routeID=route, simulator=self.simulator,
                                             initial_speed=speed, initial_position=position, initial_lane_id=lane_id), add_to_vlist=False, add_to_sumo=True)
        return {"speed": speed, "position": position}

    @staticmethod
    def generate_random_vehicle():
        """Generate a random vehicle at the beginning of the road/in FF mode.
        Returns:
            [speed, rand_position, exposure_freq]: [description]
        """
        random_number = np.random.uniform()
        idx = bisect.bisect_left(conf.speed_CDF, random_number)
        speed = conf.v_to_idx_dic.inverse[idx]
        rand_position = round(np.random.uniform(
            conf.random_veh_pos_buffer_start, conf.random_veh_pos_buffer_end))
        return speed, rand_position

    @staticmethod
    def sample_CF_FF_mode():
        """Randomly choose the Cf or FF mode to generate vehicles.

        Returns:
            str: Mode ID.
        """
        random_number_CF = np.random.uniform()
        if random_number_CF > conf.CF_percent:
            return "FF"
        else:
            return "CF"

    def generate_FF_vehicle(self, back_speed_position=None, direction=1):
        """Generate vehicle in FF mode, back_vehicle is needed.

        Args:
            back_vehicle (dict, optional): speed and position. Defaults to None.

        Returns:
            dict: Return the given speed and position.
        """
        if back_speed_position is not None:
            rand_speed, rand_position = self.generate_random_vehicle()
            pos_generate = back_speed_position["position"] + \
                (conf.ff_dis + rand_position + conf.LENGTH)*direction
            return {"speed": rand_speed, "position": pos_generate}
        else:
            raise Exception(
                "Warning: generating FF vehicle with no back vehicle")

    def generate_CF_vehicle(self, back_speed_position=None, direction=1):
        """Generate vehicles in the CF mode.

        Args:
            back_speed_position (dict optional): Speed and position. Defaults to None.

        Returns:
            dict: Vehicle information including speed, and position.
        """
        if back_speed_position["speed"] < conf.v_low:
            presum_list = conf.presum_list_forward[conf.v_to_idx_dic[conf.v_low]]
        else:
            presum_list = conf.presum_list_forward[conf.v_to_idx_dic[int(
                back_speed_position["speed"])]]
        random_number = np.random.uniform()
        r_idx, rr_idx = divmod(bisect.bisect_left(
            presum_list, random_number), conf.num_rr)
        try:
            r, rr = conf.r_to_idx_dic.inverse[r_idx], conf.rr_to_idx_dic.inverse[rr_idx]
        except:
            if back_speed_position["speed"] > 35:
                r, rr = 50, -2
            else:
                r, rr = 50, 2
        if r <= 0:
            r = r + conf.r_high
        speed = back_speed_position["speed"] + rr
        # speed limit is 20 ~ 40 m/s
        if speed > 40:
            speed = 40
        position = back_speed_position["position"] + \
            (r + conf.LENGTH)*direction
        return {"speed": speed, "position": position}
    
    def safe_av_bv_generation(self, cav_speed, cav_position, bv_speed, bv_position):
        min_distance = 15
        time_headway = 1
        required_bv_position = cav_position + min_distance + max(0, time_headway * (cav_speed - bv_speed))
        return bv_speed, max(bv_position, required_bv_position)

    def _terminate_check(self):
        collision = tuple(set(self.simulator.detected_crash()))
        reason = None
        stop = False
        additional_info = {}
        if bool(collision) and "CAV" in collision:
            reason = {1: "CAV and BV collision"}
            stop = True
            additional_info = {"collision_id": collision}
        elif "CAV" not in self.vehicle_list:
            reason = {2: "CAV leaves network"}
            stop = True
        elif self.simulator.detect_vehicle_num() == 0:
            reason = {3: "All vehicles leave network"}
            stop = True
        elif self.simulator.get_vehicle_position("CAV")[0] > 800.0: # This value is different for different experiments
            reason = {4: "CAV reaches 800 m"}
            stop = True
        if stop:
            self.episode_info["end_time"] = self.simulator.get_time(
            )-self.simulator.step_size
        return reason, stop, additional_info
