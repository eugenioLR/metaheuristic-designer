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
    def __init__(self, pop_init: Initializer, operator_list: List[Operator], params: Union[ParamScheduler, dict] = None, name: str = "PCRO-SL"):
        super().__init__(pop_init, operator_list, params=params, name=name)
        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)

    def update_params(self, progress):
        """
        Updates the parameters and the operators
        """

        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)

        super().update_params(progress)