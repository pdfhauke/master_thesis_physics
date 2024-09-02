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
from artsLUT_fitting_fitting_process import *
from sklearn.model_selection import train_test_split


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

temperaturepoint = 11
specienumber = 1
frequencynumber = 18
pressure = 11
temperature = LUT.t_pert[temperaturepoint] + LUT.t_ref[temperaturepoint]

x = np.log10(LUT.p_grid)
print("length p_grid", len(LUT.p_grid))
#x = LUT.p_grid
y = np.log10(LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]) 
#y = LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]
#y = 1/(LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]) 
#sigmoid y
#y = 1/(np.exp(-LUT.xsec[temperaturepoint,specienumber,frequencynumber,:])+1) 

#try out 1 dimensioal fitting of temmperature at fixed low pressure 
#x = np.array(LUT.t_pert + LUT.t_ref)
#y = np.log10(LUT.xsec[:,specienumber,frequencynumber,pressure]) 

#make a sub-p_grid for the fitting process and for testing how good the fit is
#to avoid overfitting 

p_grid_train, p_grid_test, y_train, y_test = train_test_split(LUT.p_grid, LUT.xsec[temperaturepoint,specienumber,frequencynumber,:], train_size = 0.7, test_size = 0.3)

print("p_grid trian", p_grid_train)
print("y_test", y_train)
print("p grid ", LUT.p_grid)
print("y grid", y)


#x = p_grid_train
#y = y_train 





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

""""""
#import one of the main functions fitting_process
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
""""""














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
#poly_poly = poly_fit(x,y, 9)

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



poly_poly = poly_fit(x,y,9)
#encapsulates the quality of the polynomial fit and its order 
fitted_z_data_poly_res = poly_fit_qual(x,y,9)

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
















#here the central function that fittes and then plots fit, plot and residuals is called here 

#i = 10
#while i < 15:
#    frequencynumber = i
#    fitting_process(temperaturepoint,specienumber,frequencynumber)
#    i = i+1
def max_deviation(frequency, specie, temperature):
    frequencynumber = frequency
    specienumber = specie
    temperaturepoint = temperature
    max_mistake = fitting_process(temperaturepoint, specienumber, frequencynumber, 4, 3)
    specie_array = ['CO2', 'O3', 'CO2 and O3']
    temperaturedegree = LUT.t_pert[temperaturepoint] + LUT.t_ref[temperaturepoint]
    print("Maximal deviation for 3rd order is ", max_mistake[0], "and check same with other method: ", max_mistake[1])

    #make loop over many diffrent orders of polynomic fit and determine the absolute value of the maximal deviation between data and fit for fitting of ln(sigma)
    max_dev_poly_array = []
    max_dev_poly_rel_array = []
    j = 0
    while j < 10:
        max_mistake = fitting_process(temperaturepoint, specienumber, frequencynumber, 4, (j+1))
        max_abs_mistake = max_mistake[0]
        max_rel_mistake = max_mistake[1]
        max_dev_poly_array.append(max_abs_mistake)
        max_dev_poly_rel_array.append(100 * max_rel_mistake)
        j += 1
    print(max_dev_poly_array)



    #plot the diffrent orders of fitting to the 1dim absolute maximal deviation 
    #and for the 1dim relative deviation between polynomic fit and data 
    #fitted is ln(sigma) but plotted sigma 
    fig5, ax5 = plt.subplots(figsize=(16, 9))
    #order_array = np.arange(1,len(max_dev_poly_array),1)
    order_array = np.linspace(1,len(max_dev_poly_array),(len(max_dev_poly_array)))
    print(order_array)
    ax5.plot(order_array, max_dev_poly_array, label = f'max. abs. data fit deviation for {wvn[frequencynumber]:0.1f}cm$^{{-1}}$ for {species[specienumber]} at {temperaturedegree:0.1f} K')
    ax5.set_xlabel("order n of fit", fontsize = 25)
    ax5.set_ylabel("abs. deviation [m$^2$]", fontsize = 25)
    ax5.set_title("Maximal abs. deviation cross section data and fit at different polynomic orders", fontsize = 20)
    ax5.legend(fontsize = 20)
    ax5.yaxis.get_offset_text().set_fontsize(25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    #plt.show()
    #path1 = f"/Users/hdamerow/master_thesis/pictures Master thesis playground /1 dim fits /max. deviation sigma poly order fitting of ln(sigma)/cross_section_1dimfitsln(sigma)_abs._maximal_deviation_data_fit_polyorders{specie_array[specienumber]}_{wvn[frequencynumber]:0.1f}_{temperaturedegree:0.1f}K.png"
    #plt.savefig(path1, dpi = 300, bbox_inches='tight')

    #fitted is ln(sigma) but plotted sigma 
    fig6, ax6 = plt.subplots(figsize=(16, 9))
    #order_array = np.arange(len(max_dev_poly_rel_array))
    order_array = np.linspace(1,len(max_dev_poly_array),(len(max_dev_poly_array)))
    print(order_array)
    ax6.plot(order_array, max_dev_poly_rel_array,label = f'max. rel. data fit deviation for {wvn[frequencynumber]:0.1f}cm$^{{-1}}$ for {species[specienumber]} at {temperaturedegree:0.1f} K')
    ax6.set_xlabel("order n of fit",fontsize = 25)
    ax6.set_ylabel("rel. deviation [%]",fontsize = 25)
    ax6.set_title("Maximal rel. deviation cross section data and fit at different polynomic orders", fontsize = 20)
    ax6.legend(fontsize = 20)
    ax6.yaxis.get_offset_text().set_fontsize(25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    ax6.set_ylim([0,6])
    #plt.show()
    path2 = f"/Users/hdamerow/master_thesis/pictures Master thesis playground /1 dim fits /max. deviation sigma poly order fitting of ln(sigma)/cross_section_zoomin_1dimfitsln(sigma)_rel._maximal_deviation_data_fit_polyorders_{specie_array[specienumber]}_{wvn[frequencynumber]:0.1f}_{temperature:0.1f}K.png"
    plt.savefig(path2, dpi = 300, bbox_inches='tight')

#try out the new compact maximal deviation function
#max_deviation(6, 0, temperaturepoint)















#real central core of this script
#now iterate over the whole excel table, stored in physical_params 
#and plot all data and fits for the different frequencies in a single plot  
#now here are plotted in 3 diffrent subplots the arbitrary order polynomial fit and data of the cross section after the ln(sigma) was fitted in this polynomial order
#in the other 2 subplots the absolute and relative deviation between derived cross section fit and data are plotted separately 


#physical_params_specialrows = pd.concat([physical_params.iloc[3:5],physical_params.iloc[7:9]])
physical_params_specialrows = physical_params.iloc[9:]
#.iloc[:17]
#now iterate over the whole excel table
#and plot all data and fits for the different frequencies in a single plot  
#fig1, (ax1,ax2) = plt.subplots(nrows=2, ncols =1, sharex = True)
def cumulative_plotting(gas_species = 'CO2', degree = 3):
    degree = degree
    gas_species = gas_species 
    if gas_species == 'CO2':
        physical_params_specialrows = physical_params.iloc[:9]
        n_color = 10
    if gas_species == 'O3':
        physical_params_specialrows = physical_params.iloc[9:]
        n_color = 6 
    if gas_species == 'CO2 and O3':
        physical_params_specialrows = physical_params
        n_color = 16 

    fig1, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 9), nrows = 3, ncols =1, sharex = True)
    #n_color  = 6
    color = iter(plt.cm.rainbow(np.linspace(0,1,n_color)))
    c = 0
    #print(color)
    for index, row in physical_params_specialrows.iterrows():
        c = next(color)
        t = row['Column2']
        specie = row['specienumber ']
        frequency = row['Column1']
        peak_character = row['Column4']
        #fitting_process(t,specie,frequency,(index+1))
        #c = next(color)

        temperaturepoint = t
        specienumber = specie
        frequencynumber = frequency
        temperature = LUT.t_pert[temperaturepoint] + LUT.t_ref[temperaturepoint]
        x = np.log10(LUT.p_grid)
        #x = LUT.p_grid
        #y = np.log10(LUT.xsec[temperaturepoint,specienumber,frequencynumber,:]) 
        y = np.log10(LUT.xsec[temperaturepoint,specienumber,frequencynumber,:])
        #y = 1/(np.exp(-LUT.xsec[temperaturepoint,specienumber,frequencynumber,:])+1) 

        #x and y for training and testing with less points 
        #x = np.log10(np.sort(p_grid_train))
        #y = np.log10(np.sort(y_train)) 


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
        #try out to fit heigher order polynomials in the same way 
        poly_poly = poly_fit(x,y,degree)

        #CO2 and O3 should be plotted in different colors 
        if specie == 1:
            linestyle = '--'
        else:
            linestyle = '-'

        #markers can be used to characterize whether frequency lies at the peak center(peak_character 2 with squares), in the wing or far away from peak center (pesk character is 0 and with dot marker )
        #peak_character = 0
        if peak_character == 0:
            symbol = 'o'
        if peak_character == 2:
            symbol = 's'
        #here the derived sigma from fit and from data are plotted for the cubic fitting from ln(sigma)
        #max_deviation(frequencynumber, specienumber, temperaturepoint)
        #ax1.plot(x,d3(x), label = 'cubic fit')
        #ax1.plot(x,y,'o', label ='ln(sigma) data')
        ax1.scatter(x, 10**y,marker = symbol, label = 'sigma data', c = c, linestyle = linestyle)
        #ax1.plot(x,d3(x), label = f' {wvn[frequencynumber]:0.1f} cm$^{{-1}}$ {species[specie]} at {temperature:0.1f} K')
        #ax1.plot(x, np.exp(d3(x)), label = f' fit {wvn[frequencynumber]:0.1f} cm$^{{-1}}$ {species[specie]} at {temperature:0.1f} K', c = c, linestyle = linestyle)
        #quartic or in general polynomial fit or arbitrary order 
        ax1.plot(x, 10**(d_poly(x,degree)), label = f' fit {wvn[frequencynumber]:0.1f} cm$^{{-1}}$ {species[specie]} at {temperature:0.1f} K', c = c, linestyle = linestyle)
        print(d3(x))
        #ax1.set_ylim([-55,-21])
        ax1.set_ylabel("σ [m$^2$]", fontsize = 25)
        ax1.yaxis.get_offset_text().set_fontsize(25)
        #ax1.set_ylabel("log10(σ/σ$_0$)", fontsize = 30)
        #ax1.set_ylabel("σ [m$^2$]")
        #ax1.set_xlabel("$log10(p/p_0$)", fontsize = 30)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25) 
        ax1.legend(ncol = 3, loc = 'upper left', fontsize = 8)
        if gas_species == 'CO2':
            ax1.set_title(f"{degree}th order fit and computed data of the frequency dependent cross section of CO$_2$", fontsize = 25)
        if gas_species == 'O3':
            ax1.set_title(f"{degree}th order fit and computed data of the frequency dependent cross section of O$_3$", fontsize = 25)
        if gas_species == 'CO2 and O3':
            ax1.set_title(f"{degree}th order fit and computed data of the frequency dependent cross section of CO$_2$ and  O$_3$", fontsize = 25)
        #ax1.set_ylim([0, 5 * 10 **(-10)])
        #and O$_3$, CO$_2$ peaks

        #here the derived deviation of sigma between fit and data is plotted for the cubic fitting from ln(sigma)

        #ax1.plot(x,d3(x), label = 'cubic fit')
        #ax1.plot(x,y,'o', label ='ln(sigma) data')
        #ax2.plot(x,d3(x)- y,marker = symbol, label = 'deviation sigma fit and data', c = c, linestyle = linestyle)
        #quartic poly fit
        ax2.plot(x,(10**(d_poly(x,degree))- 10**(y)),marker = symbol, label = 'deviation sigma fit and data', c = c, linestyle = linestyle)
        #ax1.plot(x,d3(x), label = f' {wvn[frequencynumber]:0.1f} cm$^{{-1}}$ {species[specie]} at {temperature:0.1f} K')
        #ax1.plot(x, np.exp(d3(x)), label = f' fit {wvn[frequencynumber]:0.1f} cm$^{{-1}}$ {species[specie]} at {temperature:0.1f} K', c = c, linestyle = linestyle)
        #print(d3(x))
        #ax1.set_ylim([-55,-21])
        ax2.set_ylabel("σ$_{fit}$-σ$_{data}$ [m$^2$]", fontsize = 25)
        ax2.yaxis.get_offset_text().set_fontsize(25)
        #ax1.set_ylabel("log10(σ/σ$_0$)", fontsize = 30)
        #ax1.set_ylabel("σ [m$^2$]")
        #ax2.set_xlabel("$log10(p/p_0$)", fontsize = 30)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25) 
        #ax2.legend(ncol = 3, loc = 'upper left', fontsize = 15)
        ax2.set_title("Abs. deviation between fit and computed data", fontsize = 25)

        #ax1.plot(x,d3(x), label = 'cubic fit')
        #ax1.plot(x,y,'o', label ='ln(sigma) data')
        #ax3.plot(x,((d3(x)- y)/y) * 100,marker = symbol, label = 'deviation sigma fit and data', c = c, linestyle = linestyle)
        #ax3.plot(x,d3(x), label = f' {wvn[frequencynumber]:0.1f} cm$^{{-1}}$ {species[specie]} at {temperature:0.1f} K')
        #ax1.plot(x, np.exp(d3(x)), label = f' fit {wvn[frequencynumber]:0.1f} cm$^{{-1}}$ {species[specie]} at {temperature:0.1f} K', c = c, linestyle = linestyle)
        #print(d3(x))
        #quartic fit or polynomial fit of arbitrary order 
        ax3.plot(x,((10**(d_poly(x,degree))- 10**(y))/10**(y)) * 100,marker = symbol, label = 'deviation sigma fit and data', c = c, linestyle = linestyle)
        #ax1.set_ylim([-55,-21])
        ax3.set_ylabel("rel. deviation [%]", fontsize = 25)
        ax3.yaxis.get_offset_text().set_fontsize(25)
        #ax1.set_ylabel("log10(σ/σ$_0$)", fontsize = 30)
        #ax1.set_ylabel("σ [m$^2$]")
        ax3.set_xlabel("$log10(p/p_0$)", fontsize = 25)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25) 
        #ax3.legend(ncol = 3, loc = 'upper left', fontsize = 15)
        ax3.set_title("Rel. deviation between fit and computed data", fontsize = 25)

        if index > 14:
            break
        print(frequency)
    #plt.show()
    pathc = f"/Users/hdamerow/master_thesis/pictures Master thesis playground /1 dim fits /cumulative fits 1 dim ln(sigma) CO2 and O3/Cross_section_{degree}th_order_fit_data_{gas_species}from_fitted_ln(sigma)_with_rel_abs_deviation.png"
    plt.savefig(pathc, dpi = 300, bbox_inches='tight')

    

#i =1
#while i < 10:
#   cumulative_plotting('CO2', i)
#   cumulative_plotting('O3', i)
#   cumulative_plotting('CO2 and O3', i)
    
#    i = i+1
#i = 3
cumulative_plotting('CO2 and O3', 9)
print("length p_grid", len(LUT.p_grid))
#max_deviation(60, 1, 11)

originarray = np.arange(20)

train_array, test_array = train_test_split(originarray, test_size = 0.25, train_size = 0.75)
print(test_array)

print(p_grid_train)
print(y_test)
print(LUT.p_grid)
print(y)

