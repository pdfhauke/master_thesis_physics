import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import lstsq


#now do a general polynomic fit  
def poly_fit(x, y, n):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),(n+1)))
    i = 1
    while i < (n + 1):
        M[:,i] = x**i
        i = i + 1 
    #M[:,2] = 
    #M[:,3] = x**3

    #do the linear fit
    poly_poly, res, s, sing = lstsq(M,y)
    #print("linear fit poly of order ",n, "is", poly)
    return poly_poly 

def poly_fit_qual(x, y, n):
    #tell which fit you want to do 
    #here a linear fit at x 
    M = np.ones((len(x),(n+1)))
    i = 1
    while i < (n + 1):
        M[:,i] = x**i
        i = i + 1 
    #M[:,2] = 
    #M[:,3] = x**3

    #do the linear fit
    poly_poly, res, s, sing = lstsq(M,y)
    return res, n 

#poly_poly = poly_fit(x,y, 5)

def d_poly(x, n):
    i = 0
    interres = 0
    d = 0
    n = n
    while i < (n+1):
        interres = poly_poly[i] * x**i 
        d = d + interres
        i = i + 1
    return d 

