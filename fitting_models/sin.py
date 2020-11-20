import numpy as np
import lmfit
from lmfit import Model, Parameter, Parameters


def model(coordinates: np.ndarray, A: float, omega: float,
          phi: float, offset: float) -> float:
    return A * np.sin(coordinates * omega + phi) + offset


# NOTE: I think we should allow guessing each parameter individually
def guess_A(coordinates: np.ndarray, data: np.ndarray) -> Parameter:
    amp = (np.max(data) - np.min(data)) / 2
    param = Parameter("A", amp)
    return param

def guess_offset(coordinates: np.ndarray, data: np.ndarray) -> Parameter:
    offset_ = np.mean(data)
    param = Parameter("offset", offset_)
    return param

def guess_allParams(coordinates: np.ndarray, data: np.ndarray) -> Parameters:
    model_func = Model(model)
    params = model_func.make_params()
    params['A'] = guess_A(coordinates, data)
    params['offset'] = guess_offset(coordinates, data)

    return params
