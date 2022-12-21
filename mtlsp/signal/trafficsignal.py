class Signal:
    def __init__(self):
        self.id = None,
        self._recent_observation = None

    # @property
    # def observation(self):
    #     if not self._recent_observation or self._recent_observation.time_stamp != self.simulator.get_time(): #if the recent observation exists and the recent observation is not updated at the current timestamp, update the recent observation
    #         self._recent_observation = self._get_observation()
    #     return self._recent_observation

    def _get_observation(self):
        pass

if __name__ == "main":
    signal1 = Signal()
    signal_observation = signal1._get_observation()
