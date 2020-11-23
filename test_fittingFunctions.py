from fitting_models import Sinusoidal
import numpy as np
import matplotlib.pyplot as plt
from Fitter import Fitter
from lmfit import Model, Parameter, Parameters

# -------- create data set  -------------------
coordinates = np.arange(60)
A = 2
phi = 0.5
omega = 0.5
offset = 10
data = A*np.sin(omega * coordinates + phi) + offset + A*np.random.rand(len(coordinates))


#-------- test model package---------------
params = Sinusoidal.guess_allParams(coordinates, data)


#-------- test model Fitter---------------
test = Fitter(Sinusoidal)
params = Parameters()
result = test.fit(coordinates, data, params)


plt.figure()
plt.plot(coordinates, data)
plt.plot(coordinates, result.best_fit)