import math
from mtlsp.controller.vehicle_controller.idmcontroller import IDMController
from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from envs.nde import *
from controller.nadecontroller import NADEBackgroundController
from controller.nadeglobalcontroller import NADEBVGlobalController
from nadeinfoextractor import NADEInfoExtractor


class NADE(NDE):
    def __init__(self,
                 BVController=NADEBackgroundController,
                 cav_model="RL"
                 ):
        if cav_model == "IDM":
            cav_controller = IDMController
        else:
            raise ValueError("Unknown AV controller!")
        super().__init__(
            AVController=cav_controller,
            BVController=BVController,
            AVGlobalController=DummyGlobalController,
            BVGlobalController=NADEBVGlobalController,
            info_extractor=NADEInfoExtractor
        )
        self.initial_weight = 1

    # @profile
    def _step(self):
        """NADE subscribes all the departed vehicles and decides how to control the background vehicles.
        """
        super()._step()
