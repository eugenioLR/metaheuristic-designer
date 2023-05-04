from __future__ import annotations
import numpy as np
import random
from ..Initializer import Initializer
from ..Individual import Individual


class LambdaInitializer(Initializer):
    """
    Abstract population initializer class
    """

    def __init__(self, generator: callable, popSize: int = 1, encoding: Encoding = None):
        self.popSize = popSize
        self.generator = generator

        if encoding is None:
            encoding = DefaultEncoding()
        self.encoding = encoding
    
    def generate_random(self, objfunc: ObjectiveFunc) -> Individual:
        """
        Generates a random individual
        """

        return Individual(objfunc, self.generator(), encoding=self.encoding)