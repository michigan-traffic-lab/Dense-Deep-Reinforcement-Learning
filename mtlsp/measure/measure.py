from abc import ABC
from mtlsp.logger.infoextractor import MprismInfoExtractor


class Measure(ABC):
    def evaluate(self):
        pass
    
class MprismMeasure(Measure):
    def __init__(self):
        # parameters heres
        pass
            
    def evaluate(self, snapshot_info):
        return 1

