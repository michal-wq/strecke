import numpy as np
import scipy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Zielfunktion
fun = lambda x: x[0] + x[1]

# Nebenbedingungen
cons = ({'type': 'ineq', 'fun': lambda x:  1 - x[0]**2 - x[1]**2})

# Optimierung
res = minimize(fun, (1, 1), method='SLSQP', constraints=cons)

# Ausgabe
print(f'Die Funktion wird an stellen {res["x"]} mit einem Wert {fun(res["x"])} minimiert.')