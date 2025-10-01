"""
========================================================
Author: Sevan Harootonian
Affiliation: Princeton University
========================================================
"""

import random
from frozendict import frozendict

from msdm.core.distributions import DictDistribution

from functions.functions import solve_mdp, create_mdp

class Mentee(object):
    def __init__(
        self,
        believed_mdp_params
    ):
        self.believed_mdp_params = frozendict(believed_mdp_params)
    
    def do_task(self, start=None, rng=random):
        plan_res = solve_mdp(self.believed_mdp_params)
        mdp = create_mdp(self.believed_mdp_params)
        traj = plan_res.policy.run_on(mdp, initial_state=start, rng=rng)
        full_traj = traj.state_traj + (traj.action_traj[-1], )
        return full_traj
    