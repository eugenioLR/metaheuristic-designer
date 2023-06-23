from __future__ import annotations
import numpy as np
import random
from typing import Union, List
from copy import copy, deepcopy
from ...Individual import Individual
from ...Operators import OperatorReal, OperatorMeta
from ...SurvivorSelection import SurvivorSelection
from ...ParentSelection import ParentSelection
from ..StaticPopulation import StaticPopulation
from ...ParamScheduler import ParamScheduler
from ...Algorithm import Algorithm
from .CRO_SL import CRO_SL


class PCRO_SL(CRO_SL):
    """
    Probabilistic Coral Reef Optimization with Substrate Layers.

    Published in:
    - Pérez-Aracil, Jorge, et al. "New Probabilistic, Dynamic Multi-Method Ensembles for Optimization Based on the CRO-SL." Mathematics 11.7 (2023): 1666.

    Original implementation in https://github.com/jperezaracil/PyCROSL/
    """

    def __init__(self, pop_init: Initializer, operator_list: List[Operator], params: Union[ParamScheduler, dict] = None, name: str = "PCRO-SL"):
        super().__init__(pop_init, operator_list, params=params, name=name)
        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)

    def update_params(self, progress=0):
        """
        Updates the parameters and the operators
        """

        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)

        super().update_params(progress)