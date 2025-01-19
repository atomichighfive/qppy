from typing import Optional, Dict, Tuple

import numpy as np
import numba as nb

from qppy.solvers import Solver
import qppy as qp

class InteriorPointSolver(Solver):
    def __init__(self, instance: qp.Instance):
        self.instance = instance

        self.Q, self.c, self.A, self.b = self.instance.get_standard_form()

        self.num_variables = self.Q.shape[0]
        self.num_linear_constraints = self.A.shape[0]