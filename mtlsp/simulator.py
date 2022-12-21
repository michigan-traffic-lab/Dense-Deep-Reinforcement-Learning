import os, sys
# sys.path.append("/home/haoweis/usr/lib64/python3.6/site-packages")
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")
import conf.conf as conf
libsumo_flag = (not conf.simulation_config["gui_flag"])
if libsumo_flag:
    import libsumo as traci
else:
    import traci
import traci.constants as tc
import sumolib
import numpy as np
import math
from mtlsp.network.trafficnet import TrafficNet
import time
def dummy_function (sim):
    return


class Simulator(object):    
    '''
        Simulator deals everything about synchronization of states between SUMO and python script
    '''
    def __init__(self,
        sumo_config_file_path,
        sumo_net_file_path,
        sumo_control_state = True,
        pre_step = dummy_function,
        num_tries = 10,
        step_size = 0.1,
        lc_duration = 1,
        action_step_size = 0.1,
        sublane_flag=True,
        gui_flag=False,
        track_cav=False,
        input=None,
        input_path=None,
        experiment_path=None,
        output=None,
        config={"max_obs_range": 115}):
        self.env = None
        self.sumo_net_file_path = sumo_net_file_path
        self.sumo_config_file_path = sumo_config_file_path
        self.gui_flag = gui_flag
        self.track_cav = track_cav
        self.lc_duration = lc_duration
        self.action_step_size = action_step_size
        self.config = config
        # if gui_flag:
        #     self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo-gui')
        # else:
        #     self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin/sumo')
        if gui_flag:
            self.sumo_binary = 'sumo-gui'
        else:
            self.sumo_binary = 'sumo'
        self.sumo_control_state = sumo_control_state
        self.started = False
        self.pre_step = pre_step
        self.current_time_steps = 0
        self.num_tries = num_tries
        self.step_size = step_size
        self.sublane_flag = sublane_flag        
        self.init_flag = False
        self.input_path = input_path
        self.input = input
        self.experiment_path = experiment_path
        self.output = output
        self.output_filename = 'all'
        self.split_run_flag = False
        self.episode = 0
        self.stop_reason = None

    def track_vehicle_gui(self, vehID="CAV"):
        """Track specific vehicle in GUI.

        Args:
            vehID (str, optional): Vehicle ID. Defaults to "CAV".
        """        
        traci.gui.trackVehicle(viewID='View #0', vehID=vehID)

    def get_road_ID(self, vehID):
        """Get ID of the road where the vehicle drives on.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            str: Road ID.
        """        
        return traci.vehicle.getRoadID(vehID)

    def changeTarget(self, vehID, edgeID):
        traci.vehicle.changeTarget(vehID, edgeID)

    def bind_env(self, env):
        """Combine the environment with the simulator

        Args:
            env (Environment): Simulation environment
        """        
        self.env = env
        self.env.sumo_net_file_path = self.sumo_net_file_path
        self.env.net = TrafficNet(self.sumo_net_file_path)
        env.simulator = self
    
    def detected_crash(self):
        """Detect the crash happened in the last time step.

        Returns:
            bool: True if a collision happenes in the simulation. False if no collision happens.
        """        
        colli = traci.simulation.getCollidingVehiclesIDList()
        return colli
    
    def detect_vehicle_num(self):
        """Determine the vehicle number in the simulation.

        Returns:
            int: Number of vehicles.
        """        
        return traci.simulation.getMinExpectedNumber()

    def start(self):
        """Start SUMO simulation or initialize environment.
        """        
        if self.sumo_control_state:
            sumoCmd = [self.sumo_binary, "-c", self.sumo_config_file_path, "--step-length", str(self.step_size), "--random", "--collision.mingap-factor", "0", "--collision.action", "warn"]
            if self.sublane_flag:
                sumoCmd += ["--lateral-resolution", "0.25"]
            elif self.step_size < self.lc_duration:
                sumoCmd += ["--lanechange.duration", str(self.lc_duration)]
            if self.output is not None:
                print(self.output)
                if self.split_run_flag:
                    self.output_filename = str(self.episode)
                if "traj" in self.output:
                    traj_output_filename = self.output_filename + ".traj.xml"
                    traj_output_path = os.path.join(self.experiment_path, traj_output_filename)
                    sumoCmd += ["--amitran-output", traj_output_path]
                if "fcd_all" in self.output:
                    fcd_output_filename = self.output_filename + ".fcd_all.xml"
                    fcd_output_path = os.path.join(self.experiment_path, "crash", fcd_output_filename)
                    sumoCmd += ["--fcd-output", fcd_output_path, "--fcd-output.acceleration"]
                if "fcd" in self.output:
                    fcd_output_filename = self.output_filename + ".fcd.xml"
                    fcd_output_path = os.path.join(self.experiment_path, "crash", fcd_output_filename)
                    sumoCmd += ["--fcd-output", fcd_output_path, "--fcd-output.acceleration", "--device.fcd.explicit", "CAV", "--device.fcd.radius", "200"]
                if "lc" in self.output:
                    lc_output_filename = self.output_filename + ".lc.xml"
                    lc_output_path = os.path.join(self.experiment_path, lc_output_filename)
                    sumoCmd += ["--lanechange-output", lc_output_path]
                    # if self.sublane_flag:
                    #     sumoCmd += ["--lanechange-output.started", "--lanechange-output.ended"]
                if "collision" in self.output:
                    collision_output_filename = self.output_filename + ".collision.xml"
                    collision_output_path = os.path.join(self.experiment_path, collision_output_filename)
                    sumoCmd += ["--collision-output", collision_output_path]    
            if libsumo_flag:
                traci.start(sumoCmd)
            else:
                traci.start(sumoCmd, numRetries = self.num_tries)
        else:
            self.env.initialize()
        self.started = True

    def traci_step(self, duration):
        """Simulation steps forwards.

        Args:
            duration (float): Step length in seconds.
        """        
        sim_step = traci.simulation.getTime()+duration
        traci.simulationStep(step=sim_step)

    def get_cav_travel_distance(self):
        """Get the travel distance of CAV.

        Returns:
            float: Travel distance.
        """        
        return traci.vehicle.getDistance("CAV")
    
    # @profile
    def step(self):
        """Make a simulation step.
        """        
        self.pre_step(self)
        self.env.step()
        if self.sumo_control_state:
            sim_step = traci.simulation.getTime()+self.action_step_size
            traci.simulationStep(step=sim_step)
        self.current_time_steps += 1
        self.env._check_vehicle_list()

    # @profile
    def soft_run(self):
        """Run the simulation on an existing simulation environment, we will delete all the vehicles and re-gererate them, then run the simulation.
        """
        self.env.initialize()
        traci.simulationStep()
        self.env._check_vehicle_list()
        while True:
            stop, reason, _ = self.env.terminate_check()
            if stop:
                self.stop_reason = reason
                break
            self.step()

    def plain_traci_step(self):
        traci.simulationStep()
    
    # @profile
    def run(self, episode):
        """Run the specific episode.

        Args:
            episode (int): Episode number.
        """        
        self.split_run_flag = True
        self.episode = episode
        self.start()
        self.soft_run()
        self.stop()

    def stop(self):
        """Close SUMO simulation.
        """        
        if self.started:
            traci.close()
            self.started = False
    
    def get_time(self):
        """Get current simulation time in SUMO.

        Returns:
            float: Simulation time in s.
        """       
        return traci.simulation.getTime()
    
    def get_available_lanes_id(self, edge_id, veh_type='passenger'):
        """Get available lanes for the specific edge and vehicle type. 

        Args:
            edge_id (str): Edge ID.
            veh_type (str, optional): Vehicle type. Defaults to 'passenger'.

        Returns:
            list(str): List of lane ID.
        """        
        lane_num = self.get_edge_lane_number(edge_id)
        lanes_id = []
        for i in range(lane_num):
            lane_id = edge_id+"_"+str(i)
            if veh_type not in self.get_lane_disallowed(lane_id):
                lanes_id.append(lane_id)
        return lanes_id

    def get_available_lanes(self, edge_id=None):
        """Get the available lanes in the sumo network

        Returns:
            list(sumo lane object): Possible lanes to insert vehicles
        """        
        sumo_net = sumolib.net.readNet(self.sumo_net_file_path)
        if edge_id == None:
            sumo_edges = sumo_net.getEdges()
        else:
            sumo_edges = [sumo_net.getEdge(edge_id)]
        available_lanes = []
        for edge in sumo_edges:
            for lane in edge.getLanes():
                available_lanes.append(lane)
        return available_lanes
    
    def get_edge_dist(self, first_edge_id, first_lane_position, second_edge_id, second_lane_position):
        """Get distance between two edge position.

        Args:
            first_edge_id (str): Edge ID of the first edge.
            first_lane_position (float): Lane position on the first edge.
            second_edge_id (str): Edge ID of the second edge.
            second_lane_position (float): Lane position on the second edge.

        Returns:
            float: Distance between two positions.
        """        
        return traci.simulation.getDistanceRoad(first_edge_id, first_lane_position, second_edge_id, second_lane_position, True)

    def get_edge_length(self, edgeID):
        """Get the length of the edge.

        Args:
            edgeID (str): Edge ID.

        Returns:
            float: Edge length.
        """        
        lane_id = edgeID+"_0"
        lane_length = self.get_lane_length(lane_id)
        return lane_length
    
    def get_lane_length(self, laneID):
        """Get the lane length.

        Args:
            laneID (str): Lane ID.

        Returns:
            float: Lane length.
        """        
        return traci.lane.getLength(laneID)

    def get_lane_width(self, laneID):
        """Get lane width.

        Args:
            laneID (str): Lane ID.

        Returns:
            float: Lane width in m.
        """        
        return traci.lane.getWidth(laneID)

    def get_lane_links(self, laneID):
        """Get successor lanes together with priority, open and foe for each link.

        Args:
            laneID (str): Lane ID.

        Returns:
            list: Successor lane information.
        """        
        return traci.lane.getLinks(laneID)

    def get_lane_disallowed(self, laneID):
        """Get disallowed vehicle class of the lane.

        Args:
            laneID (str): Lane ID.

        Returns:
            list(str): Disallowed vehicle class, such as "passenger".
        """        
        return traci.lane.getDisallowed(laneID)
    
    def get_route_edges(self, routeID):
        """Get edges of the route.

        Args:
            routeID (str): Route ID.

        Returns:
            list(str): A list of edge ID.
        """        
        return traci.route.getEdges(routeID)

    def _add_vehicle_to_sumo (self, v, typeID='DEFAULT_VEHTYPE', initial_position=None, initial_lane_id=None, depart=None, departLane='first', departPos='base', departSpeed='0', arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='', line='', personCapacity=0, personNumber=0):
        """Generate a vehicle in SUMO network.

        Args:
            v (Vehicle): Vehicle object of the added vehicle, including vehicle ID, route, initial speed, and lane index.
            typeID (str, optional): Vehicle type. Defaults to 'DEFAULT_VEHTYPE'.
            initial_position (float, optional): Initial position where the vehicle should enter the network. Defaults to None.
            initial_lane_id (int, optional): Initial lane index of the inserted vehicle. Defaults to None.
            depart (float, optional): Time step at which the vehicle should enter the network. Defaults to None.
            departLane (str, optional): Lane on which the vehicle should be inserted. Defaults to 'first'.
            departPos (str, optional): Position at which the vehicle should enter the net. Defaults to 'base'.
            departSpeed (str, optional): Speed with which the vehicle should enter the network. Defaults to '0'.
            arrivalLane (str, optional): Lane at which the vehicle should leave the network. Defaults to 'current'.
            arrivalPos (str, optional): Position at which the vehicle should leave the network. Defaults to 'max'.
            arrivalSpeed (str, optional): Speed with which the vehicle should leave the network. Defaults to 'current'.
            fromTaz (str, optional): Traffic assignment zone where the vehicle should enter the network. Defaults to ''.
            toTaz (str, optional): Traffic assignment zones where the vehicle should leave the network. Defaults to ''.
            line (str, optional): A string specifying the id of a public transport line which can be used when specifying person rides. Defaults to ''.
            personCapacity (int, optional): Number of all seats of the added vehicle. Defaults to 0.
            personNumber (int, optional): Number of occupied seats when the vehicle is inserted. Defaults to 0.
        """
        if libsumo_flag:
            traci.vehicle.add(v.id, v.routeID, typeID=typeID, departSpeed=str(v.initial_speed))
        else:
            traci.vehicle.add(v.id, v.routeID, typeID=typeID, depart=depart, departLane=departLane, departPos=departPos, departSpeed=v.initial_speed, arrivalLane=arrivalLane, arrivalPos=arrivalPos, arrivalSpeed=arrivalSpeed, fromTaz=fromTaz, toTaz=toTaz, line=line, personCapacity=personCapacity, personNumber=personNumber)
        if v.initial_lane_id is not None:
            traci.vehicle.moveTo(v.id, v.initial_lane_id, v.initial_position)
        # if v.initial_speed != 0:
        #     traci.vehicle.setSpeed(v.id, v.initial_speed)
    
    def _delete_all_vehicles_in_sumo(self):
        """Delete all vehicles in the network.
        """         
        for vehID in traci.vehicle.getIDList():
            traci.vehicle.remove(vehID)

    def delete_vehicle(self, vehID):
        """Delete a vehicle in SUMO simulation.

        Args:
            vehID (str): Vehicle ID.
        """
        if (vehID in self.env.av_list) or (vehID in self.env.bv_list):
            traci.vehicle.remove(vehID)
        else:
            print(f"WARNING: You are deleting vehicle id {vehID} that is not in the current environment!")

    def set_vehicle_emegency_deceleration(self, vehID, decel):
        traci.vehicle.setEmergencyDecel(vehID, decel)

    @staticmethod
    def subscribe_vehicle_ego(vehID):
        """Subscribe to store vehicle's ego information.

        Args:
            vehID (str): Vehicle ID.
        """
        traci.vehicle.subscribe(vehID, [tc.VAR_LENGTH, tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION3D, tc.VAR_EDGES, tc.VAR_LANEPOSITION, tc.VAR_LANEPOSITION_LAT, tc.VAR_SPEED_LAT, tc.VAR_ROAD_ID, tc.VAR_ACCELERATION])

    @staticmethod
    def subscribe_vehicle_surrounding(vehID, max_obs_range=120):
        """Subscribe to store vehicle's ego and surrounding information.

        Args:
            vehID (str): Vehicle ID.
        """
        Simulator.subscribe_vehicle_ego(vehID)
        traci.vehicle.subscribeContext(vehID, tc.CMD_GET_VEHICLE_VARIABLE, max_obs_range, [tc.VAR_LENGTH, tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION3D, tc.VAR_ROAD_ID, tc.VAR_ACCELERATION])
        traci.vehicle.addSubscriptionFilterLanes([-2,-1,0,1,2], noOpposite=True, downstreamDist=max_obs_range, upstreamDist=max_obs_range)
    
    @staticmethod
    def subscribe_vehicle_all_information(vehID, max_obs_range=120):
        """Subscribe to store vehicle's complete information.

        Args:
            vehID (str): Vehicle ID.
        """
        Simulator.subscribe_vehicle_ego(vehID)
        traci.vehicle.subscribeContext(vehID, tc.CMD_GET_VEHICLE_VARIABLE, max_obs_range, [tc.VAR_LENGTH, tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_INDEX, tc.VAR_ANGLE, tc.VAR_POSITION3D, tc.VAR_EDGES, tc.VAR_LANEPOSITION, tc.VAR_LANEPOSITION_LAT, tc.VAR_SPEED_LAT, tc.VAR_ROAD_ID, tc.VAR_ACCELERATION])
        traci.vehicle.addSubscriptionFilterLanes([-2,-1,0,1,2], noOpposite=True, downstreamDist=max_obs_range, upstreamDist=max_obs_range)
    
    def unsubscribe_vehicle(self, vehID):
        """Unsubscribe the vehicle information.

        Args:
            vehID (str): Vehicle ID.
        """        
        traci.vehicle.unsubscribe(vehID)

    def get_vehicle_class(self, vehID):
        """Get vehicle class.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            str: Abstract vehicle class, such as "passenger".
        """        
        return traci.vehicle.getVehicleClass(vehID)

    def get_vehicle_context_subscription_results(self, vehID):
        """Get subscription results of the context information.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            dict: Context subscription results.
        """        
        return traci.vehicle.getContextSubscriptionResults(vehID)

    def get_vehicle_min_expected_number(self):
        """Get the vehicle number in the simulation plus the number of vehicles waiting to start.

        Returns:
            int: Number of vehicles in the simulation and vehicles waiting to start.
        """        
        return traci.simulation.getMinExpectedNumber()

    def get_vehicle_speedmode(self, vehID):
        """Get speed mode of the vehicle.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            int: Speed mode.
        """        
        return traci.vehicle.getSpeedMode(vehID)

    def get_vehicle_length(self, vehID):
        """Get vehicle length.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Vehicle length in m.
        """
        return traci.vehicle.getLength(vehID)

    def get_vehicle_mingap(self, vehID):
        """Get vehicle minimum gap.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Vehicle minimum gap in m.
        """
        return traci.vehicle.getMinGap(vehID)

    def get_vehicle_acc(self, vehID):
        """Get the acceleration in m/s^2 of the named vehicle within the last step.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Vehicle acceleration [m/s^2].
        """        
        return traci.vehicle.getAcceleration(vehID)

    def get_vehicle_maxacc(self, vehID):
        """Get vehicle maximum acceleration.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Maximum acceleration in m/s^2.
        """        
        return traci.vehicle.getAccel(vehID)

    def get_vehicle_could_change_lane(self, vehID, direction):
        """Check whehther the vehicle could change lane in the specific direction.

        Args:
            vehID (str): Vehicle ID.
            direction (int): 1 represents "left" and -1 represents "right".

        Returns:
            bool: Whehther the vehicle chould change lane.
        """        
        return traci.vehicle.couldChangeLane(vehID, direction)

    def get_vehicle_lane_position(self, vehID):
        """Get the lane position of the vehicle.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Lane position in m.
        """        
        return traci.vehicle.getLanePosition(vehID)

    def get_vehicle_lane_adjacent(self, vehID, direction):
        """Get whether the vehicle is allowed to drive on the adjacent lane.

        Args:
            vehID (str): Vehicle ID.
            direction (int): 1 represents left, while -1 represents right.

        Returns:
            bool: Whether the vehicle can drive on the specific lane.
        """
        if direction not in [-1,1]:
            raise ValueError("Unknown direction input:"+str(direction))        
        lane_index = self.get_vehicle_lane_index(vehID)
        new_lane_index = lane_index+direction
        edge_id = self.get_vehicle_roadID(vehID)
        lane_num = self.get_edge_lane_number(edge_id)
        if new_lane_index < 0 or new_lane_index >=lane_num:
            # Adjacent lane does not exist.
            return False
        new_lane_id = edge_id+"_"+str(new_lane_index)
        veh_class = self.get_vehicle_class(vehID)
        disallowed = self.get_lane_disallowed(new_lane_id)
        return not veh_class in disallowed

    def get_vehicle_lane_index(self, vehID):
        """Get vehicle lane index.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            int: Lane index.
        """        
        return traci.vehicle.getLaneIndex(vehID)

    def get_edge_lane_number(self, edgeID):
        """Get lane number of the edge.

        Args:
            edgeID (str): Edge ID.

        Returns:
            int: Lane number.
        """        
        return traci.edge.getLaneNumber(edgeID)

    def get_vehicle_lane_number(self, vehID):
        """Get lane number of the edge where the vehicle is driving on.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            int: Lane number.
        """        
        return self.get_edge_lane_number(self.get_vehicle_roadID(vehID))

    def get_vehicle_lateral_lane_position(self, vehID):
        """Get the lateral offset of the vehicle.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Lateral offset related to the lane's center.
        """        
        return traci.vehicle.getLateralLanePosition(vehID)
    
    def get_vehicle_maxdecel(self, vehID):
        """Get vehicle maximum deceleration.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Maximum deceleration.
        """
        return traci.vehicle.getDecel(vehID)

    def get_vehicle_maneuver_pdf(self, vehID):
        """Get vehicle maneuver possibility distribution.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            array(float): Possibility distribution of vehicle maneuver.
        """        
        pdf_tuple = traci.vehicle.getNDDProb(vehID) # tuple of strings
        pdf_array = np.zeros(len(pdf_tuple),dtype=float) # array of floats
        for i in range(len(pdf_tuple)):
            pdf_array[i] = float(pdf_tuple[i])
        return pdf_array

    def get_vehicle_roadID(self, vehID):
        """Get road ID where the vehicle is driving on.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            str: Road ID.
        """        
        return traci.vehicle.getRoadID(vehID)

    def get_vehicle_type(self, vehID):
        """Get vehicle type ID.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            str: Type ID.
        """        
        return traci.vehicle.getTypeID(vehID)

    def get_vehicle_laneID(self, vehID):
        """Get lane ID where the vehicle is driving on.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            str: Lane ID.
        """        
        return traci.vehicle.getLaneID(vehID)

    def get_vehicle_lane_width(self, vehID):
        """Get lane width where the vehicle is driving on.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Lane width in m.
        """        
        laneID = self.get_vehicle_laneID(vehID)
        return self.get_lane_width(laneID)
    
    def get_vehicle_distance_to_edge(self, veh_id, edge_id, edge_position):
        second_edge_id = self.get_vehicle_roadID(veh_id)
        first_edge_id = edge_id
        second_lane_position = min(self.get_vehicle_lane_position(veh_id), self.get_edge_length(second_edge_id))
        first_lane_position = 0
        return traci.simulation.getDistanceRoad(first_edge_id, first_lane_position, second_edge_id, second_lane_position, True)

    def get_vehicles_dist_road(self, first_veh_id, second_veh_id):
        """Get the distance between two vehicles along the network.

        Args:
            first_veh_id (str): First vehicle ID.
            second_veh_id (str): Second vehicle ID.

        Returns:
            float: Vehicles distance.
        """ 
              
        first_edge_id = self.get_vehicle_roadID(first_veh_id)
        second_edge_id = self.get_vehicle_roadID(second_veh_id)
        first_lane_position = min(self.get_vehicle_lane_position(first_veh_id), self.get_edge_length(first_edge_id))
        second_lane_position = min(self.get_vehicle_lane_position(second_veh_id), self.get_edge_length(second_edge_id))
        return traci.simulation.getDistanceRoad(first_edge_id, first_lane_position, second_edge_id, second_lane_position, True)

    def get_vehicles_dist(self, first_veh_pos, second_veh_pos):
        """Get distance between two vehicles.

        Args:
            first_veh_pos (tuple(float,float)): 2D position of the first vehicle.
            second_veh_pos (tuple(float,float)): 2D position of the second vehicle.

        Returns:
            float: Longitudinal distance between two vehicles.
        """        
        return traci.simulation.getDistance2D(first_veh_pos[0],first_veh_pos[1],second_veh_pos[0],second_veh_pos[1],False,True)
    
    def get_vehicles_relative_lane_index(self, ego_vehID, front_vehID):
        """Get relative lane index for two vehicles.

        Args:
            ego_vehID (str): Ego vehicle ID.
            front_vehID (str): Front vehicle ID.

        Returns:
            int: Relative lane index.
        """        
        ego_laneID = self.get_vehicle_laneID(ego_vehID)
        ego_roadID = self.get_vehicle_roadID(ego_vehID)
        laneID = ego_laneID
        roadID = ego_roadID
        front_laneID = self.get_vehicle_laneID(front_vehID)
        front_roadID = self.get_vehicle_roadID(front_vehID)
        if front_roadID[0] == ":":
            links = self.get_lane_links(front_laneID)
            if len(links) > 1:
                print("WARNING: Can't locate vehicles "+str(ego_vehID)+" and "+str(front_vehID))
                return 3
            front_laneID = links[0][0]
            front_roadID = traci.lane.getEdgeID(front_laneID)
        front_lane_index = int(front_laneID.split('_')[-1])
        it = 0
        while it < 10:
            if front_roadID == roadID:
                lane_index = int(laneID.split('_')[-1])
                return front_lane_index-lane_index
            links = self.get_lane_links(laneID)
            # print(links)
            if len(links) > 1:
                print("WARNING: Can't locate vehicles "+str(ego_vehID)+" and "+str(front_vehID))
                return 3
            else:
                laneID = links[0][0]
                roadID = traci.lane.getEdgeID(laneID)
            it += 1
        print("WARNING: Can't find relative lane index for vehicles "+str(ego_vehID)+" and "+str(front_vehID))
        return 3

    def get_departed_vehID_list(self):
        """Get ID list of vehicles entering the network in the last time step.

        Returns:
            list(str): List of ID of vehicles entering the network in the last time step.
        """        
        return traci.simulation.getDepartedIDList()

    def get_vehicle_type_id(self, veh_id):
        return traci.vehicle.getTypeID(veh_id)

    def get_arrived_vehID_list(self):
        """Get ID list of vehicles arriving at the final edge in the last time step.

        Returns:
            list(str): List of ID of vehicles leaving the network in the last time step.
        """
        return traci.simulation.getArrivedIDList()

    def get_vehID_list(self):
        """Get ID list of vehicles currently in the network.

        Returns:
            list(str): List of ID of vehicles in the sumo network in the current time step.
        """        
        return traci.vehicle.getIDList()

    def get_ego_vehicle(self, vehID, dist = 0.0):
        """Get the information of the ego vehicle.

        Args:
            vehID (str): ID of the ego vehicle.
            dist (float, optional): Distance between two vehicles. Defaults to 0.0.

        Returns:
            dict: Necessary information of the ego vehicle, including:
                str: Vehicle ID (accessed by 'veh_id'), 
                float: Vehicle speed (accessed by 'velocity'),
                tuple(float, float): Vehicle position in X and Y (accessed by 'position'),
                int: Vehicle lane index (accessed by 'lane_index')
                float: Distance between the ego vehicle and another vehicle (accessed by 'distance').
        """        
        ego_veh = None
        if dist <= self.config["max_obs_range"]:
            ego_veh = {'veh_id':vehID}
            ego_veh['distance'] = dist
            try:
                # get ego vehicle information: a dict:
                # 66: position (a tuple); 64: velocity, 67: angle, 82: lane_index
                ego_info = traci.vehicle.getSubscriptionResults(vehID)
                ego_veh['velocity'] = ego_info[64]
                ego_veh['position'] = ego_info[66]
                ego_veh['heading'] = ego_info[67]
                ego_veh['lane_index'] = ego_info[82]
                ego_veh['position3D'] = ego_info[57]
                ego_veh["acceleration"] = ego_info[114]
            except:
                ego_veh['velocity'] = traci.vehicle.getSpeed(vehID)
                ego_veh['position'] = traci.vehicle.getPosition(vehID)
                ego_veh['heading'] = traci.vehicle.getAngle(vehID)
                ego_veh['lane_index'] = traci.vehicle.getLaneIndex(vehID)
                ego_veh['position3D'] = traci.vehicle.getPosition3D(vehID)
                ego_veh["acceleration"] = traci.vehicle.getAcceleration(vehID)
        return ego_veh

    def get_leading_vehicle(self, vehID):
        """Get the information of the leading vehicle.

        Args:
            vehID (str): ID of the ego vehicle.

        Returns:
            dict: necessary information of the leading vehicle, including:
                str: ID of the leading vehicle(accessed by 'veh_id'), 
                float: Leading vehicle speed (accessed by 'velocity'),
                tuple(float, float): Leading vehicle position in X and Y (accessed by 'position'),
                int: Leading vehicle lane index (accessed by 'lane_index')
                float: Distance between the ego vehicle and the leading vehicle (accessed by 'distance').
        """
        # get leading vehicle information: a list:
        # first element: leader id
        # second element: distance from leading vehicle to ego vehicle 
        # (it does not include the minGap of the ego vehicle)
        leader_info = traci.vehicle.getLeader(vehID, dist = self.config["max_obs_range"]) # empty leader: None
        if leader_info is None:
            return None
        else:
            r = leader_info[1] + traci.vehicle.getMinGap(vehID)
            return self.get_ego_vehicle(leader_info[0],r)

    def get_following_vehicle(self, vehID):
        """Get the information of the following vehicle.

        Args:
            vehID (str): ID of the ego vehicle.

        Returns:
            dict: necessary information of the following vehicle, including:
                str: ID of the following vehicle(accessed by 'veh_id'), 
                float: Following vehicle speed (accessed by 'velocity'),
                tuple(float, float): Following vehicle position in X and Y (accessed by 'position'),
                int: Following vehicle lane index (accessed by 'lane_index')
                float: Distance between the ego vehicle and the following vehicle (accessed by 'distance').
        """
        # get following vehicle information: a list:
        # first element: follower id
        # second element: distance from ego vehicle to following vehicle 
        # (it does not include the minGap of the following vehicle)
        follower_info = traci.vehicle.getFollower(vehID, dist = self.config["max_obs_range"]) # empty follower: ('',-1) 
        if follower_info[1] == -1:
            return None
        else:
            r = follower_info[1] + traci.vehicle.getMinGap(follower_info[0])
            return self.get_ego_vehicle(follower_info[0],r)

    def get_neighboring_leading_vehicle(self, vehID, dir):
        """Get the information of the neighboring leading vehicle.

        Args:
            vehID (str): ID of the ego vehicle.
            dir (str): Choose from "left" and "right".

        Returns:
            dict: necessary information of the neighboring leading vehicle, including:
                str: ID of the neighboring leading vehicle(accessed by 'veh_id'), 
                float: Neighboring leading vehicle speed (accessed by 'velocity'),
                tuple(float, float): Neighboring leading vehicle position in X and Y (accessed by 'position'),
                int: Neighboring leading vehicle lane index (accessed by 'lane_index')
                float: Distance between the ego vehicle and the neighboring leading vehicle (accessed by 'distance').
        """
        # get neighboring leading vehicle information: a list of tuple:
        # first element: leader id
        # second element: distance from leading vehicle to ego vehicle 
        # (it does not include the minGap of the ego vehicle)
        if dir == "left":
            leader_info = traci.vehicle.getNeighbors(vehID,2) # empty leftleader: len=0
        elif dir == "right":
            leader_info = traci.vehicle.getNeighbors(vehID,3) # empty rightleader: len=0
        else: 
            raise ValueError('NotKnownDirection')
        if len(leader_info) == 0:
            return None
        else:
            leader_info_list = [list(item) for item in leader_info]
            for i in range(len(leader_info)):
                leader_info_list[i][1] += traci.vehicle.getMinGap(vehID)
            sorted_leader = sorted(leader_info_list,key=lambda l:l[1])
            closest_leader = sorted_leader[0]
            return self.get_ego_vehicle(closest_leader[0],closest_leader[1])

    def get_neighboring_following_vehicle(self, vehID, dir):
        """Get the information of the neighboring following vehicle.

        Args:
            vehID (str): ID of the ego vehicle.
            dir (str): Choose from "left" and "right".

        Returns:
            dict: necessary information of the neighboring following vehicle, including:
                str: ID of the neighboring following vehicle(accessed by 'veh_id'), 
                float: Neighboring following vehicle speed (accessed by 'velocity'),
                tuple(float, float): Neighboring following vehicle position in X and Y (accessed by 'position'),
                int: Neighboring following vehicle lane index (accessed by 'lane_index')
                float: Distance between the ego vehicle and the neighboring following vehicle (accessed by 'distance').
        """
        # get neighboring following vehicle information: a list of tuple:
        # first element: follower id
        # second element: distance from ego vehicle to following vehicle 
        # (it does not include the minGap of the following vehicle)
        if dir == "left":
            follower_info = traci.vehicle.getNeighbors(vehID,0) # empty leftfollower: len=0
        elif dir == "right":
            follower_info = traci.vehicle.getNeighbors(vehID,1) # empty rightfollower: len=0
        else: 
            raise ValueError('NotKnownDirection')
        if len(follower_info) == 0:
            return None
        else:
            follower_info_list = [list(item) for item in follower_info]
            for i in range(len(follower_info)):
                follower_info_list[i][1] += traci.vehicle.getMinGap(follower_info_list[i][0])
            sorted_follower = sorted(follower_info_list,key=lambda l:l[1])
            closest_follower = sorted_follower[0]
            return self.get_ego_vehicle(closest_follower[0],closest_follower[1])
    
    def get_vehicle_speed(self, vehID):
        """Get the vehicle speed within the last step.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Vehicle speed in m/s.
        """        
        return traci.vehicle.getSpeed(vehID)

    def get_vehicle_lateral_speed(self, vehID):
        """Get the lateral speed of the vehicle.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            float: Later speed of the specified vehicle.
        """
        return traci.vehicle.getLateralSpeed(vehID)

    def get_vehicle_position(self, vehID):
        """Get the position of the vehicle.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            tuple(float, float): Position in X,Y coordinates.
        """        
        return traci.vehicle.getPosition(vehID)
    
    def get_vehicle_lane_number(self, vehID):
        """Get the number of lanes in the edge of the vehicle.

        Args:
            vehID (str): Vehicle ID.

        Returns:
            interger: Number of lanes.
        """        
        return traci.edge.getLaneNumber(traci.vehicle.getRoadID(vehID))
    
    def set_zoom(self, zoom):
        """Set the current zoom factor for the given view.

        Args:
            zoom (float): Zoom factor.
        """        
        traci.gui.setZoom(viewID='View #0', zoom=zoom)

    def set_vehicle_color(self, vehID, rgb):
        """Set the color of a vehicle to separate it from others.

        Args:
            vehID (str): Vehicle ID.
            rgb (tuple(int, int, int, int)): RGB code of the color, i.e. (255,0,0) for the color red. The fourth component (alpha) is optional.
        """        
        traci.vehicle.setColor(vehID, rgb)

    def set_vehicle_speedmode(self, vehID, speedmode=31):
        """Set the speed mode of the vehicle. This command controls how speeds set with the command setSpeed and slowDown are used. Per default, the vehicle may only drive slower than the speed that is deemed safe by the car following model and it may not exceed the bounds on acceleration and deceleration. Furthermore, vehicles follow the right-of-way rules when approaching an intersection and if necessary they brake hard to avoid driving across a red light. 

        Args:
            vehID (str): Vehicle ID.
            speedmode (int, optional): This integer is a bitset (bit0 is the least significant bit) with the following fields. Defaults to 31.
                bit0: Regard safe speed.
                bit1: Regard maximum acceleration.
                bit2: Regard maximum deceleration.
                bit3: Regard right of way at intersections.
                bit4: Brake hard to avoid passing a red light.
        """        
        traci.vehicle.setSpeedMode(vehID, speedmode)

    def set_vehicle_lanechangemode(self, vehID, lanechangemode=1621):
        """Sets how lane changing in general and lane changing requests by TraCI are performed.

        Args:
            vehID (str): Vehicle ID.
            lanechangemode (int, optional): If an external change lane command (0x13) command is in conflict with the internal request this is resolved by the current value of the vehicles lane change mode. The given integer is interpreted as a bitset (bit0 is the least significant bit) with the following fields. Defaults to 1621. 
            The default lane change mode is 0b011001010101 = 1621 which means that the laneChangeModel may execute all changes unless in conflict with TraCI. Requests from TraCI are handled urgently (with cooperative speed adaptations by the ego vehicle and surrounding traffic) but with full consideration for safety constraints.
            The lane change mode when controlling BV is 0b010001010101 = 1109 which means that the laneChangeModel may execute all changes unless in conflict with TraCI. Requests from TraCI are handled urgently without consideration for safety constraints.
                - bit1, bit0: 00 = do no strategic changes; 01 = do strategic changes if not in conflict with a TraCI request; 10 = do strategic change even if - overriding TraCI request.
                - bit3, bit2: 00 = do no cooperative changes; 01 = do cooperative changes if not in conflict with a TraCI request; 10 = do cooperative change even if overriding TraCI request.
                - bit5, bit4: 00 = do no speed gain changes; 01 = do speed gain changes if not in conflict with a TraCI request; 10 = do speed gain change even if overriding TraCI request.
                - bit7, bit6: 00 = do no right drive changes; 01 = do right drive changes if not in conflict with a TraCI request; 10 = do right drive change even if overriding TraCI request.
                - bit9, bit8:   
                    00 = do not respect other drivers when following TraCI requests, adapt speed to fulfill request;
                    01 = avoid immediate collisions when following a TraCI request, adapt speed to fulfill request;
                    10 = respect the speed / brake gaps of others when changing lanes, adapt speed to fulfill request;
                    11 = respect the speed / brake gaps of others when changing lanes, no speed adaption.
                - bit11, bit10: 00 = do no sublane changes; 01 = do sublane changes if not in conflict with a TraCI request; 10 = do sublane change even if overriding TraCI request.
        """        
        traci.vehicle.setLaneChangeMode(vehID, lanechangemode)

    def set_vehicle_max_lateralspeed(self, vehID, lat_max_v):
        """Set the maximum lateral speed of vehicle.

        Args:
            vehID (str): Vehicle ID.
            lat_max_v (float): Maximum lateral speed.
        """        
        traci.vehicle.setMaxSpeedLat(vehID, lat_max_v)

    def change_vehicle_speed(self, vehID, acceleration, duration=1.0):
        """Fix the acceleration of a vehicle to be a specified value in the specified duration.

        Args:
            vehID (str): Vehicle ID
            acceleration (float): Specified acceleration of vehicle.
            duration (float, optional): Specified time interval to fix the acceleration in s. Defaults to 1.0.
        """        
        # traci.vehicle.slowDown take duration + deltaT to reach the desired speed
        init_speed = traci.vehicle.getSpeed(vehID)
        final_speed = init_speed+acceleration*(self.step_size+duration)
        if final_speed < 0:
            final_speed = 0
        traci.vehicle.slowDown(vehID, final_speed, duration)

    def _cal_lateral_maxSpeed(self, vehID, lane_width, time=1.0):
        """Calculate the maximum lateral speed for lane change maneuver.

        Args:
            vehID (str): Vehicle ID.
            lane_width (float): Width of the lane.
            time (float, optional): Specified time interval to complete the lane change maneuver in s. Defaults to 1.0.

        Raises:
            ValueError: If the maximum lateral acceleration of the vehicle is too small, it is impossible to complete the lane change maneuver in the specified duration.

        Returns:
            float: Maximum lateral speed aiming to complete the lane change behavior in the specified time duration.
        """        
        # accelerate laterally to the maximum lateral speed and maintain
        # v^2 - b*v + c = 0
        lat_acc = float(traci.vehicle.getParameter(vehID, 'laneChangeModel.lcAccelLat'))
        b, c = lat_acc*(time), lat_acc*lane_width
        delta_power = b**2-4*c
        if delta_power >= 0:
            lat_max_v = (-math.sqrt(delta_power)+b)/2
        else:
            raise ValueError("The lateral maximum acceleration is too small.")
        return lat_max_v
    
    def _cal_lateral_distance(self, vehID, direction):
        """Calculate lateral distance to the target lane for a complete lane change maneuver.

        Args:
            vehID (str): Vehicle ID.
            direction (str): Direction, i.e. "left" and "right".

        Raises:
            ValueError: Unknown lane id.
            ValueError: Unknown lane id.
            ValueError: Unknown direction.

        Returns:
            float: Distance in m.
        """        
        origin_lane_id = traci.vehicle.getLaneID(vehID)
        edge_id = traci.vehicle.getRoadID(vehID)
        lane_index = int(origin_lane_id.split('_')[-1])
        origin_lane_width = traci.lane.getWidth(origin_lane_id)
        if direction == "left":
            target_lane_id = edge_id+"_"+str(lane_index+1)
            try:
                target_lane_width = traci.lane.getWidth(target_lane_id)
            except:
                raise ValueError("Unknown lane id: "+ target_lane_id+" in the lane change maneuver.")
            latdist = (origin_lane_width+target_lane_width)/2
        elif direction == "right":
            target_lane_id = edge_id+"_"+str(lane_index-1)
            try:
                target_lane_width = traci.lane.getWidth(target_lane_id)
            except:
                raise ValueError("Unknown lane id: "+ target_lane_id+" in the lane change maneuver.")    
            latdist = -(origin_lane_width+target_lane_width)/2
        else:
            raise ValueError("Unknown direction for lane change command")
        return latdist

    def change_vehicle_sublane_dist(self, vehID, latdist, duration):
        """Change the lateral position of the vehicle.

        Args:
            vehID (str): Vehicle ID.
            latdist (float): Desired lateral distance.
            duration (float): Change duration.
        """        
        lat_max_v = self._cal_lateral_maxSpeed(vehID, abs(latdist), duration)
        traci.vehicle.setMaxSpeedLat(vehID, lat_max_v)
        traci.vehicle.changeSublane(vehID, latdist)

    def change_vehicle_lane(self, vehID, direction, duration=1.0):
        """Force a vehicle to complete the lane change maneuver in the duration.

        Args:
            vehID (str): Vehicle ID.
            direction (str): Choose from "LANE_LEFT" and "LANE_RIGHT".
            duration (float, optional): Specified time interval to complete the lane change behavior. Defaults to 1.0.

        Raises:
            ValueError: Direction is neither "LANE_LEFT" nor "LANE_RIGHT".
        """
        if self.sublane_flag:
            latdist = self._cal_lateral_distance(vehID, direction)
            lat_max_v = self._cal_lateral_maxSpeed(vehID, abs(latdist), duration)
            traci.vehicle.setMaxSpeedLat(vehID, lat_max_v)
            traci.vehicle.changeSublane(vehID, latdist)
        else:
            if direction == "left":
                indexOffset = 1
            elif direction == "right":
                indexOffset = -1
            else:
                raise ValueError("Unknown direction for lane change command")
            traci.vehicle.changeLaneRelative(vehID, indexOffset, self.step_size)

    def change_vehicle_position(self, vehID, position, edgeID="", lane=-1, angle=-1073741824.0, keepRoute=1):
        """Move the vehicle to the given coordinates and force it's angle to the given value (for drawing).

        Args:
            vehID (str): Vehicle ID.
            position (tuple(float, float)): The specified x,y coordinates.
            edgeID (str, optional): Edge ID. Defaults to "".
            lane (int, optional): Lane index. Defaults to -1.
            angle (float, optional): angle (float, optional): Specified angle of vehicle. If the angle is set to INVALID_DOUBLE_VALUE, the vehicle assumes the natural angle of the edge on which it is driving. Defaults to -1073741824.0.
            keepRoute (int, optional): If keepRoute is set to 1, the closest position within the existing route is taken. If keepRoute is set to 0, the vehicle may move to any edge in the network but it's route then only consists of that edge. If keepRoute is set to 2, the vehicle has all the freedom of keepRoute=0 but in addition to that may even move outside the road network. Defaults to 1.
        """     
        x = position[0]
        y = position[1]
        traci.vehicle.moveToXY(vehID, edgeID, lane, x, y, angle, keepRoute)

    def subscribe_signal(self, tlsID):
        """Subscribe to the specified traffic light.
            TL_BLOCKING_VEHICLES = 37
            TL_COMPLETE_DEFINITION_RYG = 43
            TL_COMPLETE_PROGRAM_RYG = 44
            TL_CONTROLLED_JUNCTIONS = 42
            TL_CONTROLLED_LANES = 38
            TL_CONTROLLED_LINKS = 39
            TL_CURRENT_PHASE = 40
            TL_CURRENT_PROGRAM = 41
            TL_EXTERNAL_STATE = 46
            TL_NEXT_SWITCH = 45
            TL_PHASE_DURATION = 36
            TL_PHASE_INDEX = 34
            TL_PRIORITY_VEHICLES = 49
            TL_PROGRAM = 35
            TL_RED_YELLOW_GREEN_STATE = 32
            TL_RIVAL_VEHICLES = 48

        Args:
            tlsID (str): Signal ID.
        """        
        traci.trafficlight.subscribe(tlsID, [tc.TL_CURRENT_PHASE, tc.TL_PHASE_DURATION, tc.TL_RED_YELLOW_GREEN_STATE])

    def get_signal_information(self, tlsID):
        """Get the subscribed information of traffic signal in the last time step.

        Args:
            tlsID (str): Signal ID.

        Returns:
            dict: Subscribed information of the specified traffic light.
        """        
        return traci.trafficlight.getSubscriptionResults(tlsID)

    def get_tlsID_list(self):
        """Get a list of traffic light ID.

        Returns:
            list(str): List of all traffic light in the network.
        """        
        return traci.trafficlight.getIDList()

    def get_signal_state(self, tlsID):
        """Returns the named tl's state as a tuple of light definitions from rugGyYoO, for red, yed-yellow, green, yellow, off, where lower case letters mean that the stream has to decelerate.

        Args:
            tlsID (str): Signal ID.

        Returns:
            str: Current state of the specified traffic light.
        """        
        return traci.trafficlight.getRedYellowGreenState(tlsID)

    def set_signal_logic(self, tlsID, newlogic):
        """Sets a new program for the given tlsID from a Logic object.

        Args:
            tlsID (str): Signal ID.
            newlogic (Logic): New traffic light logic.
        """   
        traci.trafficlight.setProgramLogic(tlsID, newlogic)   
    