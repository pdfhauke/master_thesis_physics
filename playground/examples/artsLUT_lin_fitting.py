import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import lstsq

def lin_fit(x, y):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),2))
    M[:,1] = x
    #M[:,2] = 
    #M[:,3] = x**3

    #do the linear fit
    poly, res, s, sing = lstsq(M,y)

    print("linear fit poly", poly)
    return poly 

#function to evaluate the linear fit quality 
#res is always sum of squared residuals 
def lin_fit_qual(x, y):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),2))
    M[:,1] = x
    #M[:,2] = 
    #M[:,3] = x**3

    #do the linear fit
    poly, res, s, sing = lstsq(M,y)
    return res


# get the linear fit function 
def d(x):
    d = poly[0] +poly[1] * x
    return d 