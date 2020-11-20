from fitting_models import sin
import numpy as np
import matplotlib.pyplot as plt
from Fitter import Fitter
from lmfit import Model, Parameter, Parameters


coordinates = np.arange(60)
A = 2
phi = 0.5
omega = 0.5
offset = 10
data = A*np.sin(omega * coordinates + phi) + offset + A*np.random.rand(len(coordinates))

params = sin.guess_allParams(coordinates, data)



test = Fitter(sin)
result = test.fit(coordinates, data, ["A", "offset"])


plt.figure()
plt.plot(coordinates, data)
plt.plot(coordinates, result.best_fit)