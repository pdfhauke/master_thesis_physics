#this function does a cubic and arbitrary polynomic fit of the ln(cross_section)
#it needs as inputs "temperaturepoint, specienumber, frequencynumber,i, n" so the temperaturepoint, gas species, frequncypoint, 
# not important i, and the degree of polynomic fitting n all to be found in the arts LUT 
# it returns  the maximal desired polynomic degree deviation between data and fit as absoulte vlaue and relative value,
#relative to the data cross section at this pressure 
#it outputs additionally the maximal and minimal absolute and relative data-fit deviations for cubic fit 
#it outputs the R2 as quality indicator of fit for cubic and arbitraryly polynomic fit 
#finally it plots the cubic fit and the absolute and relative deviation between data and cubic fit 
#in a subplot,can storage it too 
# and it can write quality values into the main excel table automatically  

#now do a 1 dimensional fitting with the scipy.linalg.lstsq method

#%%
import numpy as np
from scipy.linalg import lstsq 
import matplotlib.pyplot as plt 
from pyarts import arts
from pyarts.plots import arts_lookup
import FluxSimulator as fsm
import pandas as pd 
import openpyxl as ol 
from artsLUT_lin_fitting import *
from artsLUT_quad_fittting import *
from artsLUT_cubic_fitting import *
from artsLUT_poly_fitting import *


#%%
#first set-up athmospere and the lookup-table 

#define pressure grid 
#take pressure dependent cross section of ozone at 270 K 

# set frequency grid
min_wvn = 5
#min_wvn_micro = 0.01

max_wvn = 3210
#max_wvn_micro = 10

n_freq_lw = 200
wvn = np.linspace(min_wvn, max_wvn, n_freq_lw)
#wvn_micro = np.linspace(min_wvn_micro, max_wvn_micro, n_freq_lw)
f_grid_lw = arts.convert.kaycm2freq(wvn)
species = ['CO$_2$', 'O$_3$','CH$_4$']


# setup ARTS
flux_simulator_LUT = fsm.FluxSimulator("TESTLUT_SW")
flux_simulator_LUT.ws.f_grid = f_grid_lw
flux_simulator_LUT.set_species(
    ["CO2,CO2-CKDMT252","O3","CH4"]
)


# Wide LUT
flux_simulator_LUT.get_lookuptableWide(recalc=False)
LUT = flux_simulator_LUT.ws.abs_lookup.value

#import physical parameters from table 
physical_params = pd.read_excel('/Users/hdamerow/master_thesis/1 dim fits of the p_dependent absorption cross section_LUT_O3_CO2_atm.xlsx', sheet_name = 'Sheet1')

#import same excel file as workbook 
excel_table_cross_sections = ol.load_workbook('/Users/hdamerow/master_thesis/1 dim fits of the p_dependent absorption cross section_LUT_O3_CO2_atm.xlsx')
excel_sheet_cross_sections = excel_table_cross_sections['Sheet1']

#define symbol array for plotting the different gases in the plots with different symbols 
#circle is CO2 and square is O3 
symbol_array = ['o', 's', 'v']
#array for linestyles 


#temperature = physical_params.iat[4,2]
#temperature = physical_params.at[4,'[K]']
#print("excel temperature", temperature)

temperaturepoint = 5
specienumber = 0
frequencynumber = 40
pressure = 11
temperature = LUT.t_pert[temperaturepoint] + LUT.t_ref[temperaturepoint]

x = np.log10(LUT.p_grid)
#x = LUT.p_grid
y = np.log10(LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]) 
#y = LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]
#y = 1/(LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]) 
#sigmoid y
#y = 1/(np.exp(-LUT.xsec[temperaturepoint,specienumber,frequencynumber,:])+1) 


#try out 1 dimensioal fitting of temmperature at fixed low pressure 
#x = np.array(LUT.t_pert + LUT.t_ref)
#y = np.log10(LUT.xsec[:,specienumber,frequencynumber,pressure]) 












"""""
#import the linear fitting functions lin_fit(x,y), that returns poly with the polynomails coefficients after fitting a linear x dependent function to the data y 
#import the linear fitting functions lin_fit_qual(x,y), that returns res, the residuum of the linear fitting least square process as quality indicator 
#import the linear fit construction function d(x), that returns d, the linear fit function value of x  
#import all linear fiting functions from the script "artsLUT_lin_fitting"
"""""
    
#get linear fit quality 
#function comes from the artsLUT_lin_fitting script 
polylinqual = lin_fit_qual(x,y)
"""""
#import the quadratic fitting functions quad_fit(x,y), that returns polyquad with the polynomails coefficients after fitting a quadratic x dependent function to the data y 
#import the quadratic fitting functions quad_fit_quality(x,y), that returns res, the residuum of the linear fitting least square process as quality indicator 
#import the quadratic fit construction function d2(x), that returns d, the quadratic fit function value of x  
#import all quadratic fiting functions from the script "artsLUT_quad_fittting", be careful with 3 t 
"""""
#get quad fit function object
polyquad = quad_fit(x,y)

#object encapsulates the quality of the quadratic fit 
polyquadqual = quad_fit_quality(x,y)  

"""""
#import the cubic fitting functions cube_fit(x,y), that returns polycube with the polynomails coefficients after fitting a quadratic x dependent function to the data y 
#import the cubic fitting functions cube_fit_qual(x,y), that returns res, the residuum of the linear fitting least square process as quality indicator 
#import the cubic fit construction function d3(x), that returns d, the cubic fit function value of x  
#import all cubic fiting functions from the script "artsLUT_cubic_fitting"
"""""
#oject encapsulates the cubic fitting
polycube = cube_fit(x,y)















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
    poly, res, s, sing = lstsq(M,y)

    print("linear fit poly of order ",n, "is", poly)
    return poly 

def d_poly(x, n):
    i = 0
    interres = 0
    d = 0
    while i < (n+1):
        interres = poly_poly[i] * x**i 
        d = d + interres
        i = i + 1
    return d 
#object encapsulates the polynomial fitting 
poly_poly = poly_fit(x,y, 4)

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
    poly, res, s, sing = lstsq(M,y)

    print("linear fit poly of order ",n, "is", poly)
    return res, n 



poly_poly = poly_fit(x,y,4)
#encapsulates the quality of the polynomial fit and its order 
fitted_z_data_poly_res = poly_fit_qual(x,y,4)

#outputs the derived fit of the cross section from fitting the ln(sigma) 
#not tested so far and maybe not really required  
def sigma_fit_poly(x,n):
    z = np.exp(d_poly(x,n))
    return z 


#print put the relative squared error 
#later deleting 
print("Res of the polynomial fitting with order ",fitted_z_data_poly_res[1], "is ",fitted_z_data_poly_res[0])


#only short testing od the general polynomial fit
#later deleting
#plt.plot(x, d_poly(x,4))
plt.plot(x, y, 'o')
plt.show()


#now try to do the plotting and fiting at a special frequency and temperature as a extra function 

#temperaturepoint = 5
#specienumber = 1
#frequencynumber = 62
#pressure = 11
#i = 4















#i has here no meaning 
def fitting_process(temperaturepoint, specienumber, frequencynumber,i, n):

    temperaturepoint = temperaturepoint
    specienumber = specienumber
    frequencynumber = frequencynumber
    x = np.log10(LUT.p_grid)
    #x = LUT.p_grid
    #y = LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]/x
    y = np.log10(LUT.xsec[temperaturepoint,specienumber,frequencynumber,:])
    #y = 1/(np.exp(-LUT.xsec[temperaturepoint,specienumber,frequencynumber,:])+1) 

    #do the cubic fitting 
    def cube_fit(x, y):
        #tell which fit you want to do 
        #here a linear fit at x 
        M = np.ones((len(x),4))
        M[:,1] = x
        M[:,2] = x**2
        M[:,3] = x**3

        #do the cubic fit
        poly, res, s, sing = lstsq(M,y)

        #print(poly)
        return poly 

    #oject encapsulates the cubic fitting
    polycube = cube_fit(x,y)

    # get the cubic fit function 
    def d3(x):
        d3 = polycube[0] +polycube[1] * x +polycube[2]* x**2 + polycube[3] * x**3
        return d3 

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
        poly, res, s, sing = lstsq(M,y)
        print("linear fit poly of order ",n, "is", poly)
        return poly

    #object encapsulates the polynomial fitting 
    poly_poly = poly_fit(x,y, n)

    def d_poly(x, n):
        i = 0
        interres = 0
        d = 0
        while i < (n+1):
            interres = poly_poly[i] * x**i 
            d = d + interres
            i = i + 1
        return d 
    
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
        poly, res, s, sing = lstsq(M,y)
        print("linear fit poly of order ",n, "is", poly)
        return res,n


    #object encasulates the cubic fit quality 
    #standard deviation of all simulations 
    polycubequal = cube_fit_qual(x,y)
    print("polypolyres", poly_poly)
    polycubequal_std  = np.sqrt(polycubequal/(len(x)*(len(x)-1)))
    polycubequal_rmse = np.sqrt(polycubequal/len(x))

    #object encasulates the general polynomic fit quality 
    #standard deviation of all simulations 
    #here for polynom of 4th order
    polypolyqual = poly_fit_qual(x,y,n)
    print("polypolyqual", polypolyqual)
    polypolyqual_std  = np.sqrt(polypolyqual[0]/(len(x)*(len(x)-1)))
    polypolyqual_rmse = np.sqrt(polypolyqual[0]/len(x))
    
    #process to compute maximal and minimal absolute and realtive deviation between data and fit for the cubic fitted function 
    #array with the residuals between trignomically fitted and data of precise simulation
    #implement for sigma and not ln(sigma) maximal and minimal devoiation 
    d_residuals = d3(x) - y 
    d_residuals_relative = np.array(d_residuals)/abs(y)
    #maximal and minimal absolute deviation of fit and data 
    d_residuals = np.array(d_residuals)
    d_residuals_abs = abs(d_residuals)
    min_residual = np.min(d_residuals_abs) 
    max_residual = np.max(d_residuals_abs)
    #maximal and minimal relative deviation 
    #be careful with the case when you have many same deviations and with the sign 
    index_min_rel = np.argwhere(d_residuals_abs == min_residual)
    index_min_rel = index_min_rel[0][0]
    #min_residual_relative = min_residual/y[index_min_rel]
    min_residual_relative = d_residuals[index_min_rel]/abs(y[index_min_rel])
    index_max_rel = np.argwhere(d_residuals_abs == max_residual)
    index_max_rel = index_max_rel[0][0]
    #max_residual_relative = max_residual/y[index_max_rel]
    max_residual_relative = d_residuals[index_max_rel]/abs(y[index_max_rel])

    #process to compute maximal and minimal absolute and realtive deviation between data and fit for the polynomically fitted function 
    #array with the residuals between polynomically fitted and data of precise simulation
    #polynomic order n for fit 
    n = n
    d_poly_residuals = 10**(d_poly(x,n)) - 10**(y) 
    d_poly_residuals_relative = np.array(d_poly_residuals)/abs(10**(y))
    #maximal and minimal absolute deviation of fit and data 
    d_poly_residuals = np.array(d_poly_residuals)
    d_poly_residuals_abs = abs(d_poly_residuals)
    min_poly_residual = np.min(d_poly_residuals_abs) 
    max_poly_residual = np.max(d_poly_residuals_abs)
    #maximal and minimal relative deviation 
    #be careful with the case when you have many same deviations and with the sign 
    index_poly_min_rel = np.argwhere(d_poly_residuals_abs == min_poly_residual)
    index_poly_min_rel = index_poly_min_rel[0][0]
    #min_residual_relative = min_residual/y[index_min_rel]
    min_poly_residual_relative = d_poly_residuals[index_poly_min_rel]/abs(np.exp(y[index_poly_min_rel]))
    index_poly_max_rel = np.argwhere(d_poly_residuals_abs == max_poly_residual)
    index_poly_max_rel = index_poly_max_rel[0][0]
    #max_residual_relative = max_residual/y[index_max_rel]
    max_poly_residual_relative = d_poly_residuals_abs[index_poly_max_rel]/abs(10**(y[index_poly_max_rel]))


    #the best indicator is the relative squared error (R^2 error) for a regression fit, that sould be at least 0.95
    #here omly for cubic fitting 
    mean_vector = np.mean(y)*y/y
    r_squared = 1 - polycubequal/(np.sum((y - mean_vector)**2))
    print("mean vector", np.mean(y)*y/y)

    #the best indicator is the relative squared error (R^2 error) for a regression fit, that sould be at least 0.95
    #here for general polynomial fitting 
    mean_vector = np.mean(y)*y/y
    r_squared_poly = 1 - polypolyqual[0]/(np.sum((y - mean_vector)**2))
    #print("mean vector", np.mean(y)*y/y)
    print("The R2 error for a polynomial of the", polypolyqual[1], "order is", r_squared_poly)


    #plot data and fits 
    fig1, (ax1,ax2) = plt.subplots(nrows=2, ncols =1, sharex = True)
    #fig1, ax1 = plt.subplots(nrows=1, ncols =1)
    #ax1.plot(x,d(x), label = 'linear fit')
    #ax1.plot(x,d2(x), label = 'quadratic fit')
    ax1.loglog(x,d3(x), label = 'cubic fit')
    #ax1.semilogx(x,y,'o', label ='data')
    ax1.legend()
    #fig1, ax2 = plt.subplots()
    ax2.semilogx(x,d_residuals, label = 'absolute residuals of cubic fitting and data', color = 'red')
    ax2.legend(loc= 'lower left')
    ax3 = ax2.twinx()
    #ax3.loglog(x,100*d_residuals_relative, label = 'relative residuals of cubic fitting and data in %', color = 'green') 
    ax1.set_ylabel("$log10(p/p_0$)")
    ax1.set_xlabel("P [Pa]")
    ax2.set_xlabel("P [Pa]")
    ax1.set_ylabel("σ [m$^2$]")
    ax2.set_ylabel("σ [m$^2$]")
    #ax1.set_xlabel("T [K]")
    #ax2.set_xlabel("T [K]")
    #sigma is cross section 
    #ax1.set_ylabel("log10(σ/σ$_0$)")
    #ax2.set_ylabel("log10(σ/σ$_0$)")
    #ax1.set_xlabel("$log10(p/p_0$)")
    #ax2.set_xlabel("$log10(p/p_0$)")
    ax3.set_ylabel("[%]")
    ax3.tick_params(axis= 'y', labelcolor = 'green')
    ax3.legend(loc = 'upper left')
    #plt.ylabel("cross section [$m^2$]")
    #plt.legend()
    #plt.show()
    test = d3(x)
    print("sigmoid-plot",test[10])


    #give important additional information about physics of this set up 
    temperature = LUT.t_pert[temperaturepoint] + LUT.t_ref[temperaturepoint]
    freqkays = wvn[frequencynumber]
    species = ['CO2', 'O$_3','CH4']
    print("temperature: ",temperature)
    print("frequency: ", freqkays)
    print("species: ", species[specienumber])
    print("pressure: ", LUT.p_grid[pressure])
    #print("temperature array", LUT.t_pert, LUT.t_ref)
    #print("frequency array: ", wvn)

    #output the fitting quality numbers res from the lstsq fitting of different orders 
    #res is always sum of squared residuals 
    print("linear fit quality: ", polylinqual)
    print("quadratic fit quality: ", polyquadqual)
    print("cubic fit quality res: ", polycubequal)
    print("cubic fit quality rmse: ", polycubequal_rmse)
    print("cubic fit quality standard deviation: ", polycubequal_std)
    print("minimal absolute deviation between data and fit", d_residuals[index_min_rel], "this are ", min_residual_relative*100,"%" )
    print("maximal absolute deviation between data and fit", d_residuals[index_max_rel], "this are ", max_residual_relative*100,"%" )
    print("R$^2$: ", r_squared)
    #print("minimal residual", min_residual_relative*100 )
    print("index of minimal residual", index_min_rel )

    #put all relevant quality indices into one table for the cubic fitting 
    fit_quality_table = np.array([[species[specienumber], freqkays, temperature, polycubequal,polycubequal_rmse,r_squared, d_residuals[index_min_rel], min_residual_relative*100,d_residuals[index_max_rel],max_residual_relative*100]])
    index_fit_quality = ['cubic fit']
    columns_fit_quality = ['gas','frequency[cm$^(-1)$','temperature','cubic fit quality res', 'cubic fit quality rmse','R$^2$','minimal absolute deviation data and fit','minimal rel. deviation data and fit', 'maximal absolute deviation data and fit', 'maximal     relative deviation data and fit']
    fit_quality_table = pd.DataFrame(fit_quality_table,index_fit_quality,columns_fit_quality)
    print(fit_quality_table)

    #write the relevant parameters into the excel table 
    i = i + 1
    rmse_index = 'J' + str(i)
    excel_sheet_cross_sections[rmse_index] = polycubequal_rmse
    max_res_index = 'L' + str(i)
    excel_sheet_cross_sections[max_res_index] = max_residual_relative*100
    min_res_index = 'N' + str(i)
    excel_sheet_cross_sections[min_res_index] = min_residual_relative*100
    r2_res = 'O' + str(i)
    excel_sheet_cross_sections[r2_res] = r_squared*100

    #excel_table_cross_sections.save('/Users/hdamerow/master_thesis/1 dim fits of the p_dependent absorption cross section_LUT_O3_CO2_atm.xlsx')


    #storage the plot with presure dependance of absorption cross section fit and simlated data 
    #metadata =[temperature,freqkays]
    #path = '/Users/hdamerow/master_thesis/pictures Master thesis playground /1 dim fits /einsdim_1to3_order_with_uncertainty_fit_0.753cm^(-1)_240K_CO2.png'
    #plt.savefig(path)
    #save the temperature fitted plots 
    #path_temp = '/Users/hdamerow/master_thesis/pictures Master thesis playground /1 dim fits /einsdim_1to3_order_fit_1003cm^(-1)_CO2_temperature_fit.png'
    #plt.savefig(path_temp)
    return max_poly_residual,max_poly_residual_relative
