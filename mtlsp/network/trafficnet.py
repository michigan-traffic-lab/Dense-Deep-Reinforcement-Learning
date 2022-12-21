import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import sumolib

class TrafficNet:
    def __init__(self, sumo_net_file_path):
        self.sumo_net_file_path = sumo_net_file_path
        self.sumo_net = sumolib.net.readNet(self.sumo_net_file_path)
    
    def get_available_lanes_ids(self):
        """Get the available lanes ids in the sumo network

        Returns:
            list(str object): Possible lanes to insert vehicles
        """
        return [lane.getID() for lane in self.get_available_lanes()]


    def get_available_lanes(self):
        """Get the available lanes in the sumo network

        Returns:
            list(sumo lane object): Possible lanes to insert vehicles
        """        
        sumo_edges = self.sumo_net.getEdges()
        available_lanes = []
        for edge in sumo_edges:
            for lane in edge.getLanes():
                available_lanes.append(lane)
        return available_lanes
