from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from .nddcontroller import NDDController
from mtlsp.controller.vehicle_controller.controller import Controller
BASE_NDD_CONTROLLER = NDDController
class NDDBVGlobalController(DummyGlobalController):
    def __init__(self, env, veh_type="BV"):
        super().__init__(env, veh_type)
        self.control_vehicle_set = set()

    # @profile
    def step(self):
        """If there are CAVs in the network, reset the state of all vehicles, update the subscription and decide the next action for each vehicle.
        """
        if self.apply_control_permission():
            self.reset_control_and_action_state()
            self.update_subscription(controller=BASE_NDD_CONTROLLER)
            for veh_id in self.controllable_veh_id_list:
                vehicle = self.env.vehicle_list[veh_id]
                vehicle.controller.step()
                vehicle.update()
        else:
            self.control_vehicle_set = set()

    # @profile
    def update_subscription(self, controller=NDDController):
        """To improve the computational efficiency, only the surrounding vehicles of CAV are subscribed.
        """
        sim = self.env.simulator
        CAV = self.env.vehicle_list["CAV"]

        context_vehicle_set = set(CAV.observation.context.keys())
        if self.control_vehicle_set != context_vehicle_set:
            for veh_id in context_vehicle_set:
                if veh_id not in self.control_vehicle_set:
                    sim.subscribe_vehicle_surrounding(veh_id)
                    self.env.vehicle_list[veh_id].install_controller(
                        controller())
            for veh_id in self.control_vehicle_set:
                if veh_id not in context_vehicle_set and veh_id in self.env.vehicle_list.keys():
                    sim.unsubscribe_vehicle(veh_id)
                    self.env.vehicle_list[veh_id].install_controller(
                        Controller())
            self.control_vehicle_set = context_vehicle_set
        for veh_id in self.env.vehicle_list.keys()-["CAV"]:
            if veh_id not in context_vehicle_set and self.env.vehicle_list[veh_id].controller.type != "DummyController":
                sim.unsubscribe_vehicle(veh_id)
                self.env.vehicle_list[veh_id].install_controller(Controller())
