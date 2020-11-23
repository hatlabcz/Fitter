import lmfit
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from lmfit import Model, Parameter, Parameters

class Fitter:
    def __init__(self, fitting_module):
        self.fitting_module = fitting_module
        self.model = Model(fitting_module.model)

    def fit(self, coordinates: np.ndarray, data: np.ndarray,
            guess_params:List[str]=[]):
        params =  self.fitting_module.default_parameters()
        guess_allParams = self.fitting_module.guess_allParams(coordinates, data)
        for gp in guess_params:
            params[gp] = guess_allParams[gp]
        print (params)
        result = self.model.fit(data, params, coordinates=coordinates)
        return result

