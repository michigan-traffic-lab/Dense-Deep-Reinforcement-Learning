from abc import ABC

class InfoExtractor(ABC):
    def __init__(self, env):
        self.env = env

    def add_initialization_info(self):
        pass

    def get_snapshot_info(self, control_info):
        pass

    def get_terminate_info(self, stop, reason, additional_info):
        pass

