import numpy as np 
import matplotlib.pyplot as plt 
from scipy.linalg import lstsq

#do the cubic fitting 
def cube_fit(x, y):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),4))
    M[:,1] = x
    M[:,2] = x**2
    M[:,3] = x**3

    #do the cubic fit
    polycube, res, s, sing = lstsq(M,y)

    #print(poly)
    return polycube 

#evaluate the cubic fit quality 
#res is always sum of squared residuals 
def cube_fit_qual(x, y):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),4))
    M[:,1] = x
    M[:,2] = x**2
    M[:,3] = x**3

    #do the cubic fit
    polycube, res, s, sing = lstsq(M,y)
    return res

# get the cubic fit function 
def d3(x):
    d3 = polycube[0] +polycube[1] * x +polycube[2]* x**2 + polycube[3] * x**3
    return d3 