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
from .PCRO_SL import PCRO_SL


class DPCRO_SL(CRO_SL):
    def __init__(self, pop_init: Initializer, operator_list: List[Operator], params: Union[ParamScheduler, dict] = {}, name: str = "DPCRO-SL"):
        super().__init__(pop_init, operator_list, params=params, name=name)

        self.dyn_method = params["dyn_method"]
        self.dyn_metric = params["dyn_metric"]
        self.dyn_steps = params["dyn_steps"]
        self.prob_amp = params["prob_amp"]

        self.operator_idx = random.choices(range(len(self.operator_list)), k=self.maxpopsize)
        self.operator_weight = [1/len(operator_list)]*len(operator_list)

        self.operator_data = [[] for i in operator_list]

        if self.dyn_method == "success":
            self.operator_data = [0 for i in self.operator_data]
            self.larva_count = [0 for i in operator_list]
        elif self.dyn_method == "diff":
            self.operator_metric_prev = [0 for i in operator_list]
        
        self.operator_w_history = []
        self.op_steps = 0
        self.operator_metric = [0]*len(operator_list)
        self.operator_history = []
    
    def _operator_metric(self, data):
        result = 0

        # Choose what information to extract from the data gathered
        if len(data) > 0:
            data = sorted(data)
            if self.dyn_metric == "best":
                result = max(data)
            elif self.dyn_metric == "avg":
                result = sum(data)/len(data)
            elif self.dyn_metric == "med":
                if len(data) % 2 == 0:
                    result = (data[len(data)//2-1]+data[len(data)//2])/2
                else:
                    result = data[len(data)//2]
            elif self.dyn_metric == "worse":
                result = min(data)
        
        return result
    
    def _operator_probability(self, values):
        # Normalization to avoid passing big values to softmax 
        weight = np.array(values)
        weight_sum = np.abs(weight).sum()
        if weight_sum != 0:
            weight = weight/weight_sum
        else:
            weight = weight/(weight_sum+1e-5)
        
        # softmax to convert to a probability distribution
        exp_vec = np.exp(weight)
        amplified_vec = exp_vec**(1/self.prob_amp)
        
        # if there are numerical error default repeat with a default value
        if (amplified_vec == 0).any() or not np.isfinite(amplified_vec).all():
            if not self.prob_amp_warned:
                print("Warning: the probability amplification parameter is too small, defaulting to prob_amp = 1")
                self.prob_amp_warned = True
            prob = exp_vec/exp_vec.sum()
        else:
            prob = amplified_vec/amplified_vec.sum()

        # If probabilities get too low, equalize them
        if (prob <= 0.02/len(values)).any():
            prob += 0.02/len(values)
            prob = prob/prob.sum()

        return prob
    
    def _evaluate_operators(self):
        metric = 0
        
        # take reference data for the calculation of the difference of the next evaluation
        if self.dyn_method == "diff":
            full_data = [d for subs_data in self.operator_data for d in subs_data]
            metric = self._operator_metric(full_data)
        
        # calculate the value of each operator with the data gathered
        for idx, s_data in enumerate(self.operator_data):
            if self.dyn_method == "success":

                # obtain the rate of success of the larvae
                if self.larva_count[idx] > 0:
                    self.operator_metric[idx] = s_data[0]/self.larva_count[idx]
                else:
                    self.operator_metric[idx] = 0

                # Reset data for nex iteration
                self.operator_data[idx] = [0]
                self.larva_count[idx] = 0

            elif self.dyn_method == "fitness" or self.dyn_method == "diff":

                # obtain the value used in the evaluation of the operator 
                self.operator_metric[idx] = self._operator_metric(s_data)

                # Calculate the difference of the fitness in this generation to the previous one and
                # store the current value for the next evaluation
                if self.dyn_method == "diff":
                    self.operator_metric[idx] =  self.operator_metric[idx] - self.operator_metric_prev[idx]
                    self.operator_metric_prev[idx] = metric
                
                # Reset data for next iteration
                self.operator_data[idx] = []
    
    def _generate_substrates(self, progress=0):
        n_operators = len(self.operator_list)

        if progress > self.op_steps/self.dyn_steps:
            self.op_steps += 1
            self._evaluate_operators()

        # Assign the probability of each operator
        if self.dynamic:
            self.operator_weight = self._operator_probability(self.operator_metric)
            self.operator_w_history.append(self.operator_weight)
        
        # Choose each operator with the weights chosen
        self.operator_list = random.choices(range(n_operators), 
                                            weights=self.operator_weight, k=self.size)

        # save the evaluation of each operator
        self.operator_history.append(np.array(self.operator_metric))
    
    def perturb(self, parent_list, objfunc, progress, history):
        offspring = []
        for idx, indiv in enumerate(parent_list):
            
            # Select operator
            op_idx = self.operator_idx[idx]
            
            op = self.operator_list[op_idx]

            # Apply operator
            new_indiv = op(indiv, parent_list, objfunc, self.best, self.pop_init)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Collect data about each operator
            if self.dyn_method == "fitness" or self.dyn_method == "diff":
                self.substrate_data[s_idx].append(new_indiv.get_fitness())

            # Add to offspring list
            offspring.append(new_indiv)

        # Update best solution
        current_best = max(offspring, key=lambda x: x.fitness)
        if self.best.fitness < current_best.fitness:
            self.best = current_best

        return offspring
    
    def select_individuals(self, population, offspring, progress=0, history=None):
        return self.selection_op(population, offspring)

    def update_params(self, progress):
        self._generate_substrates(progress)
        super().update_params(progress)