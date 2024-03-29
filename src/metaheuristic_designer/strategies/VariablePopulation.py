from __future__ import annotations
import random
from ..ParamScheduler import ParamScheduler
from ..selectionMethods import (
    SurvivorSelection,
    ParentSelection,
    SurvivorSelectionNull,
    ParentSelectionNull,
)
from ..SearchStrategy import SearchStrategy
from ..Operator import Operator


class VariablePopulation(SearchStrategy):
    """
    Population-based optimization strategy where the number of individuals generated is different from the size of the population
    """

    def __init__(
        self,
        initializer: Initializer,
        operator: Operator,
        parent_sel: ParentSelection = None,
        survivor_sel: SurvivorSelection = None,
        n_offspring: int = None,
        params: ParamScheduler | dict = {},
        name: str = "Variable Population Evolution",
    ):
        self.params = params
        self.operator = operator

        if n_offspring is None and initializer is not None:
            n_offspring = initializer.pop_size
        self.n_offspring = n_offspring

        self.best = None

        super().__init__(
            initializer,
            parent_sel=parent_sel,
            survivor_sel=survivor_sel,
            params=params,
            name=name,
        )

    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, new_initializer):
        self.n_offspring = new_initializer.pop_size
        self._initializer = new_initializer

    def perturb(self, parent_list, objfunc, **kwargs):
        offspring = []

        while len(offspring) < self.n_offspring:
            # Apply operator
            indiv = random.choice(parent_list)
            new_indiv = self.operator(indiv, parent_list, objfunc, self.best, self.initializer)
            new_indiv.genotype = objfunc.repair_solution(new_indiv.genotype)
            new_indiv.speed = objfunc.repair_speed(new_indiv.speed)

            # Add to offspring list
            offspring.append(new_indiv)

        return offspring

    def update_params(self, **kwargs):
        super().update_params(**kwargs)

        progress = kwargs["progress"]

        if isinstance(self.operator, Operator):
            self.operator.step(progress)

        if isinstance(self.survivor_sel, SurvivorSelection):
            self.survivor_sel.step(progress)

        if self.param_scheduler:
            self.param_scheduler.step(progress)
            self.params = self.param_scheduler.get_params()
