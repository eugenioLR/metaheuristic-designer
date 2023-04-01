from __future__ import annotations
import random
import numpy as np
from typing import Union
from ..Individual import Individual
from ..ParamScheduler import ParamScheduler
from ..Algorithm import Algorithm


class HillClimb(Algorithm):
    """
    Search strtategy example, HillClimbing
    """
    
    def __init__(self, perturb_op: Operator, params: Union[ParamScheduler, dict]={}, name: str="HillClimb"):
        """
        Constructor of the Example search strategy class
        """

        self.perturb_op = perturb_op
        self.iterations = params["iters"] if "iters" in params else 1

        super().__init__(name, popSize=1)

    
    def perturb(self, indiv_list, objfunc, progress=0, history=None):
        """
        Performs a step of the algorithm
        """
        
        result = []

        for indiv in indiv_list:            
            for i in range(self.iterations):                
                # Perturb individual
                new_indiv = self.perturb_op(indiv, indiv_list, objfunc, self.best)
                new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
    
                # Store best vector for individual
                new_indiv.store_best(indiv)

                # If it improves the previous solution keep it
                if new_indiv.fitness > indiv.fitness:
                    indiv = new_indiv        
            
            result.append(indiv)

        curr_best = max(result, key=lambda x: x.fitness)
        if curr_best.fitness > self.best.fitness:
            self.best = curr_best
                
        return result
    
    
    def update_params(self, progress):
        """
        Updates the parameters of each component of the algorithm
        """

        self.perturb_op.step(progress)

    
    