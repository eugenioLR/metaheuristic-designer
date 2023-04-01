from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
from .ParamScheduler import ParamScheduler


class Operator(ABC):
    """
    Abstract Operator class
    """

    def __init__(self, method: str, params: Union[ParamScheduler, dict], name=None):
        """
        Constructor for the Operator class
        """

        self.method = method.lower()
        self.param_scheduler = None

        if name is None:
            name = method
        self.name = name

        if params is None:

            # Default parameters
            self.params = {
                "F": 0.5, 
                "Cr": 0.8,
                "Par":0.1,
                "N":5,
                "method": "gauss",
                "temp_ch":10,
                "iter":20,
                "Low":-10,
                "Up":10
            }
        else:
            if "method" in params:
                params["method"] = params["method"].lower()

            if isinstance(params, ParamScheduler):
                self.param_scheduler = params
                self.params = self.param_scheduler.get_params()
            else:
                self.params = params
        
    

    def __call__(self, solution: Individual, population: List[Individual], objfunc: ObjectiveFunc, global_best: Individual) -> Individual:
        """
        A shorthand for calling the 'evolve' method
        """

        return self.evolve(solution, population, objfunc, global_best)
    
    
    def step(self, progress: float):
        """
        Updates the parameters of the method using a paramater scheduler if it exists
        """

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()


    @abstractmethod
    def evolve(self, solution: Individual, population: List[Individual], objfunc: ObjectiveFunc, global_best: Individual) -> Individual:
        """
        Evolves a solution with a different strategy depending on the type of substrate
        """
