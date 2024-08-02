#in this script the 1 dim leastsquares quadratic fitting functions are made available 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import lstsq

#do a quadratic fit 
def quad_fit(x, y):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),3))
    M[:,1] = x
    M[:,2] = x**2
    #M[:,3] = x**3

    #do the quad fit
    polyquad, res, s, sing = lstsq(M,y)
    return polyquad 

#evaluate the quadratic fit function, returns res 
#res is always sum of squared residuals 
def quad_fit_quality(x, y):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),3))
    M[:,1] = x
    M[:,2] = x**2
    #M[:,3] = x**3

    #do the quad fit
    polyquad, res, s, sing = lstsq(M,y)
    return res

# get the quadratic fit function 
def d2(x):
    d2 = polyquad[0] +polyquad[1] * x +polyquad[2]* x**2 
    return d2 
