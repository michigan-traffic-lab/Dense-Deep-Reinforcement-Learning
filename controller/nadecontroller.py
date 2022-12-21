import numpy as np
from .nddcontroller import NDDController
import utils
from conf import conf

BASE_NDD_CONTROLLER = NDDController
class NADEBackgroundController(BASE_NDD_CONTROLLER):
    def __init__(self):
        super().__init__(controllertype="NADEBackgroundController")
        self.weight = None
        self.ndd_possi = None
        self.critical_possi = None
        self.epsilon_pdf_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.normalized_critical_pdf_array = np.zeros(
            len(conf.ACTIONS), dtype=float)
        self.ndd_possi_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_criticality_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_challenge_array = np.zeros(len(conf.ACTIONS), dtype=float)

    def _type(self):
        return 'NADEBackgroundController'

    # @profile
    def get_NDD_possi(self):
        return np.array(self.ndd_pdf)

    # @profile
    def _sample_critical_action(self, bv_criticality, criticality_array, ndd_possi_array, epsilon=conf.epsilon_value):
        """Sample critical action of the controlled BV using the defensive importance sampling: Owen, A. B. Monte Carlo Theory, Methods and Examples. https://statweb.stanford.edu/~owen/mc/  (2013).

        Args:
            criticality_array (list(float)): List of criticality of each BV maneuver.
            bv_criticality (float): Total criticality of the studied BV.
            possi_array (list(float)): List of probability of each BV maneuver.

        Returns:
            integer: Sampled BV action index.
            float: Weight of the BV action.
            float: Possibility of the BV action.
            float: Criticality of the BV action.
        """
        normalized_critical_pdf_array = criticality_array / bv_criticality
        epsilon_pdf_array = utils.epsilon_greedy(
            normalized_critical_pdf_array, ndd_possi_array, epsilon=epsilon)
        bv_action_idx = np.random.choice(
            len(conf.BV_ACTIONS), 1, replace=False, p=epsilon_pdf_array)
        critical_possi, ndd_possi = epsilon_pdf_array[bv_action_idx], ndd_possi_array[bv_action_idx]
        weight_list = (ndd_possi_array+1e-30)/(epsilon_pdf_array+1e-30) # avoid zero division
        weight = ndd_possi/critical_possi
        self.bv_criticality_array = criticality_array
        self.normalized_critical_pdf_array = normalized_critical_pdf_array
        self.ndd_possi_array = ndd_possi_array
        self.epsilon_pdf_array = epsilon_pdf_array
        self.weight = weight.item()
        self.ndd_possi = ndd_possi.item()
        self.critical_possi = critical_possi.item()
        return bv_action_idx, weight, ndd_possi, critical_possi, weight_list

    @staticmethod
    # @profile
    def _hard_brake_challenge(v, r, rr):
        """Calculate the hard brake challenge value. 
           Situation: BV in front of the CAV, do the hard-braking.

        Args:
            v (float): Speed of BV.
            r (float): Distance between BV and CAV.
            rr (float): Range rate of BV and CAV.

        Returns:
            list(float): List of challenge for the BV behavior.
        """
        CF_challenge_array = np.zeros((len(conf.BV_ACTIONS)-2), dtype=float)
        round_speed, round_r, round_rr = utils._round_data_plain(v, r, rr)
        index = np.where(
            (conf.CF_state_value == [round_r, round_rr, round_speed]).all(1))
        assert(len(index) <= 1)
        index = index[0]
        if len(index):
            CF_challenge_array = conf.CF_challenge_value[index.item(), :]
        new_r = r + rr
        if new_r <= 2.1:
            CF_challenge_array = np.ones(
            (len(conf.BV_ACTIONS) - 2), dtype=float)
        return CF_challenge_array

    @staticmethod
    # @profile
    def _BV_accelerate_challenge(v, r, rr):
        """Assume the CAV is cutting in the BV and calculate by the BV CF

        Args:
            v (float): Speed of BV.
            r (float): Distance between BV and CAV.
            rr (float): Range rate between BV and CAV.

        Returns:
            float: Challenge of the BV behavior.
        """
        BV_CF_challenge_array = np.zeros(
            (len(conf.BV_ACTIONS) - 2), dtype=float)
        round_speed, round_r, round_rr = utils._round_data_plain(v, r, rr)

        index = np.where((conf.BV_CF_state_value == [
                         round_r, round_rr, round_speed]).all(1))
        assert(len(index) <= 1)
        index = index[0]

        if len(index):
            BV_CF_challenge_array = conf.BV_CF_challenge_value[index.item(), :]
        new_r = r + rr
        if new_r <= 2.1:
            BV_CF_challenge_array = np.ones(
            (len(conf.BV_ACTIONS) - 2), dtype=float)
        return BV_CF_challenge_array


    def Decompose_decision(self, CAV, SM_LC_prob, full_obs=None, predicted_full_obs=None, predicted_traj_obs=None):
        """calculate the criticality of the given BV

        Args:
            CAV (dict): Observation of the CAV.
            SM_LC_prob (list(float)): List of possibility for the CAV from the surrogate model.

        Returns:
            float: Total criticality of the BV.
            integer: BV action index.
            float: Weight of the BV action.
            float: Possibility of the BV action.
            float: Criticality of the BV action.
        """
        self.bv_criticality_array = np.zeros(len(conf.ACTIONS), dtype=float)
        bv_criticality, bv_action_idx, weight, ndd_possi, critical_possi, weight_list, criticality_array = - \
            np.inf, None, None, None, None, None, np.zeros(len(conf.ACTIONS), dtype=float)
        bv_id = self.vehicle.id
        bv_pdf = self.get_NDD_possi()
        bv_obs = self.vehicle.observation.information
        bv_left_prob, bv_right_prob = bv_pdf[0], bv_pdf[1]
        # If in lane change mode or lane change prob = 1, the bv will not be controlled by the D2RL agent. So we do not need to calculate the criticality.
        if not ((0.99999 <= bv_left_prob <= 1) or (0.99999 <= bv_right_prob <= 1)):
            bv_criticality, criticality_array, bv_challenge_array, risk = self._calculate_criticality(
                bv_obs, CAV, SM_LC_prob, full_obs, predicted_full_obs, predicted_traj_obs)
            self.bv_challenge_array = bv_challenge_array
        return bv_criticality, criticality_array
    
    def Decompose_sample_action(self, bv_criticality, bv_criticality_array, bv_pdf, epsilon=conf.epsilon_value):
        """ Sample the critical action of the BV.
        """
        if epsilon is None:
            epsilon = conf.epsilon_value
        bv_action_idx, weight, ndd_possi, critical_possi, weight_list = None, None, None, None, None
        if bv_criticality > conf.criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi, weight_list = self._sample_critical_action(
                bv_criticality, bv_criticality_array, bv_pdf, epsilon)
        if weight is not None:
            weight, ndd_possi, critical_possi = weight.item(
            ), ndd_possi.item(), critical_possi.item()
        return bv_action_idx, weight, ndd_possi, critical_possi, weight_list