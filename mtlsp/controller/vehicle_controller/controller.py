from abc import ABC, abstractmethod
import numpy as np
from bidict import bidict
from mtlsp.observation.observation import Observation
from mtlsp.simulator import Simulator

class Controller(ABC):
    """Controller class deal with the control of the vehicle based on observation
    """    
    def __init__(self, observation_method=None, subscription_method=None, controllertype="DummyController"):
        """Initialize a Controller object.

        Args:
            controllertype (str, optional): Type of the controller. Defaults to None.
        """        
        self.ego_info = None
        self.action = None
        self._type = controllertype
        self.observation_method = observation_method
        self.subscription_method = subscription_method
        self.control_log = {} # This will have the control log result for each controller
    
    def reset(self):
        pass

    # @profile
    def install(self):
        self.vehicle.simulator.set_vehicle_speedmode(self.vehicle.id)
        self.vehicle.simulator.set_vehicle_lanechangemode(self.vehicle.id)
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_yellow)
        self.vehicle.simulator.set_vehicle_emegency_deceleration(self.vehicle.id, 4)
        if self.subscription_method:
            self.subscription_method(self.vehicle.id)
        if self.observation_method:
            self.vehicle.observation_method = self.observation_method

    def step(self):
        pass

    @property
    def type(self):
        """Return controller type.

        Returns:
            str: Controller type.
        """        
        return self._type


class BaseController(Controller):
    longi_safety_buffer, lateral_safety_buffer = 2, 2
    v_low, v_high, r_low, r_high, rr_low, rr_high, acc_low, acc_high =20, 40, 0, 115, -10, 8, -4, 2
    acc_resolution = 0.2
    LENGTH = 5
    ACTION_STEP = 1.0
    num_acc = int(1+((acc_high-acc_low)/acc_resolution))
    CAV_acc_low, CAV_acc_high, CAV_acc_step = -4, 2, 0.2
    num_CAV_acc = int((CAV_acc_high - CAV_acc_low)/CAV_acc_step + 1)
    CAV_acc_to_idx_dic = bidict()
    for i in range(num_CAV_acc): CAV_acc_to_idx_dic[list(np.arange(CAV_acc_low, CAV_acc_high + CAV_acc_step, CAV_acc_step))[i]] = i
    acc_to_idx_dic = bidict()
    for m in range(num_acc): acc_to_idx_dic[list(np.linspace(acc_low, acc_high, num=num_acc))[m]] = m

    def __init__(self, observation_method = Observation, subscription_method=Simulator.subscribe_vehicle_ego, controllertype="BaseController"):
        super().__init__(controllertype=controllertype, observation_method=observation_method, subscription_method=subscription_method)

    def step(self):
        """Store ego vehicle information.
        """        
        self.ego_info = self.vehicle.observation.information["Ego"]
    
    def install(self):
        super().install()
        self.vehicle.simulator.set_vehicle_speedmode(self.vehicle.id, 0)
        self.vehicle.simulator.set_vehicle_lanechangemode(self.vehicle.id, 0)
        self.vehicle.simulator.set_vehicle_emegency_deceleration(self.vehicle.id, 4)

    @staticmethod
    # @profile
    def _check_longitudinal_safety(obs, pdf_array, lateral_result=None, CAV_flag=False):
        """Check longitudinal safety for vehicle.

        Args:
            obs (dict): Processed observation of the vehicle.
            pdf_array (list(float)): Old possibility distribution of the maneuvers.
            lateral_result (list(float), optional): Possibility distribution of the lateral maneuvers. Defaults to None.
            CAV_flag (bool, optional): Check whether the vehicle is the CAV. Defaults to False.

        Returns:
            list(float): New possibility distribution of the maneuvers after checking the longitudinal direction.
        """        
        ego_info = obs['Ego']
        f_veh_info = obs['Lead']
        safety_buffer = BaseController.longi_safety_buffer
        for i in range(len(pdf_array)-1, -1, -1):
            if CAV_flag:
                acc = BaseController.CAV_acc_to_idx_dic.inverse[i]
            else:
                acc = BaseController.acc_to_idx_dic.inverse[i]
            if f_veh_info is not None:
                rr = f_veh_info["velocity"] - ego_info["velocity"]
                r = f_veh_info["distance"]
                criterion_1 = rr + r + 0.5 * (BaseController.acc_low - acc)
                self_v_2, f_v_2 = max(ego_info["velocity"] + acc, BaseController.v_low), max((f_veh_info["velocity"] + BaseController.acc_low), BaseController.v_low)                
                dist_r = (self_v_2**2 - BaseController.v_low **2)/(2*abs(BaseController.acc_low))
                dist_f = (f_v_2**2 - BaseController.v_low **2)/(2*abs(BaseController.acc_low)) + BaseController.v_low * (f_v_2 - self_v_2) / BaseController.acc_low
                criterion_2 = criterion_1 - dist_r + dist_f
                if criterion_1 <= safety_buffer or criterion_2 <= safety_buffer:
                    pdf_array[i] = 0
                else:
                    break
        
        # Only set the decelerate most when none of lateral is OK.
        if lateral_result is not None:
            lateral_feasible = lateral_result[0] or lateral_result[2]
        else:            
            lateral_feasible = False       
        if np.sum(pdf_array) == 0 and not lateral_feasible:
            pdf_array[0] = 1 if not CAV_flag else np.exp(-2)
            return pdf_array
        
        if CAV_flag:
            new_pdf_array = pdf_array
        else:
            new_pdf_array = pdf_array / np.sum(pdf_array)
        return new_pdf_array

    @staticmethod
    # @profile 
    def _check_lateral_safety(obs, pdf_array, CAV_flag=False):
        """Check the lateral safety of the vehicle.

        Args:
            obs (dict): Processed information of vehicle observation.
            pdf_array (list): Old possibility distribution of the maneuvers.
            CAV_flag (bool, optional): Check whether the vehicle is the CAV. Defaults to False.

        Returns:
            list: New possibility distribution of the maneuvers after checking the lateral direction.
        """        
        CAV_observation = obs
        f0, r0 = CAV_observation["LeftLead"], CAV_observation["LeftFoll"]
        f2, r2 = CAV_observation["RightLead"], CAV_observation["RightFoll"]
        CAV_info = CAV_observation["Ego"]
        lane_change_dir = [0, 2]
        nearby_vehs = [[f0, r0], [f2, r2]]
        safety_buffer = BaseController.lateral_safety_buffer
        ### need to change when considering more than 3 lanes
        if not obs["Ego"]["could_drive_adjacent_lane_right"]:
            pdf_array[2] = 0
        elif not obs["Ego"]["could_drive_adjacent_lane_left"]:
            pdf_array[0] = 0
        for lane_index, nearby_veh in zip(lane_change_dir,nearby_vehs):
            if pdf_array[lane_index] != 0:
                f_veh,r_veh = nearby_veh[0], nearby_veh[1]
                if f_veh is not None:
                    rr = f_veh["velocity"] - CAV_info["velocity"]
                    r = f_veh["distance"]
                    dis_change = rr*BaseController.ACTION_STEP + 0.5*BaseController.acc_low*(BaseController.ACTION_STEP**2)
                    r_1 = r + dis_change # 1s
                    rr_1 = rr + BaseController.acc_low*BaseController.ACTION_STEP
        
                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0             
                    elif rr_1 < 0:
                        self_v_2, f_v_2 = max(CAV_info["velocity"], BaseController.v_low), max((f_veh["velocity"] + BaseController.acc_low), BaseController.v_low)                
                        dist_r = (self_v_2**2 - BaseController.v_low **2)/(2*abs(BaseController.acc_low))
                        dist_f = (f_v_2**2 - BaseController.v_low **2)/(2*abs(BaseController.acc_low)) + BaseController.v_low * (f_v_2 - self_v_2) / BaseController.acc_low
                        r_2 = r_1 - dist_r + dist_f                        
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0
                if r_veh is not None:
                    rr = CAV_info["velocity"] - r_veh["velocity"]
                    r = r_veh["distance"]
                    dis_change = rr*BaseController.ACTION_STEP - 0.5*BaseController.acc_high*(BaseController.ACTION_STEP**2)
                    r_1 = r + dis_change
                    rr_1 = rr - BaseController.acc_high*BaseController.ACTION_STEP
                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0                        
                    elif rr_1 < 0:
                        self_v_2, r_v_2 = min(CAV_info["velocity"], BaseController.v_high), min((r_veh["velocity"] + BaseController.acc_high), BaseController.v_high)                
                        dist_r = (r_v_2**2 - BaseController.v_low **2)/(2*abs(BaseController.acc_low))
                        dist_f = (self_v_2**2 - BaseController.v_low **2)/(2*abs(BaseController.acc_low)) + BaseController.v_low * (-r_v_2 + self_v_2) / BaseController.acc_low
                        r_2 = r_1 - dist_r + dist_f 
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0
        if np.sum(pdf_array) == 0:
            return np.array([0, 1, 0])
        
        if CAV_flag:
            new_pdf_array = pdf_array
        else:
            new_pdf_array = pdf_array / np.sum(pdf_array)        
        return new_pdf_array





# class ACMAVController(AVController):

#     def step(self):
#         super().step()
#         if self.vehicle.simulator.getRoadID(self.vehicle.id) == "-1003000.207.30":
#             self.vehicle.simulator.changeTarget(self.vehicle.id, "-1008000.289.23")
#         else:
#             self.vehicle.simulator.changeTarget(self.vehicle.id, "-1003000.207.30")