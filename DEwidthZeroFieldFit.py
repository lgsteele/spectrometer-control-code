import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import peakutils

######################################
######################################
# Start fitting function
######################################
######################################
def de(filename,zfArray): # determines D, E, and width from zero-field data
    def lor(f,a1,a2,D,E,width,offset):
        lor1 = a1*((width**2)/(np.pi*width*((f-(D+E))**2 + width**2)))
        lor2 = a2*((width**2)/(np.pi*width*((f-(D-E))**2 + width**2)))
        return lor1+lor2 + offset
    def gauss(f,a1,a2,D,E,width,offset):
        gauss1 = a1*np.exp(-(4*np.log(2)*((f-(D+E))**2))/(width**2))
        gauss2 = a2*np.exp(-(4*np.log(2)*((f-(D-E))**2))/(width**2))
        return gauss1+gauss2+offset
    freq, signal = np.loadtxt('%s.txt' % filename,\
                              delimiter=', ',skiprows=1,unpack=True)
##    signal = signal/np.amax(signal) # normalize the signal to 1

######################################
# Fitting with Lorentzian
######################################
    p0=[-3e7,-3e7,\
        zfArray[0],zfArray[1],zfArray[2],zfArray[3]]
    yguess = lor(freq,-3e7,-3e7,\
        zfArray[0],zfArray[1],zfArray[2],zfArray[3])
    coeffs, matcov = curve_fit(lor, freq, signal, p0)
    yStdErr = np.sqrt(np.diag(matcov)) # compute standard errors of fit params
##    print yStdErr
    yfit = lor(freq,coeffs[0],coeffs[1],coeffs[2],\
               coeffs[3],coeffs[4],coeffs[5])
    zfArray[0] = coeffs[2] # D
    zfArray[1] = coeffs[3] # E
    zfArray[2] = coeffs[4] # width
    zfArray[3] = coeffs[5] # offset
    zfArray[4] = yStdErr[2] # std err of D
    zfArray[5] = yStdErr[3] # std err of E
    zfArray[6] = yStdErr[4] # std err of width
    zfArray[7] = yStdErr[5] # std err of offset
##    yfitarray = np.column_stack([freq,yfit])
##    np.savetxt('Gd-248K-fitArray.txt',yfitarray,delimiter=', ',\
##           header='freq, data', comments='')
    plt.plot(freq,signal,'r-',freq,yguess,'g-',freq,yfit,'b-')
    plt.show()
    return zfArray
######################################
# Fitting with Gaussian
######################################
##    p0=[np.amax(signal),np.amax(signal),\
##        zfArray[0],zfArray[1],zfArray[2],zfArray[3]]
##    yguess = gauss(freq,np.amax(signal),np.amax(signal),\
##        zfArray[0],zfArray[1],zfArray[2],zfArray[3])
##    coeffs, matcov = curve_fit(gauss, freq, signal, p0)
##    yStdErr = np.sqrt(np.diag(matcov)) # compute standard errors of fit params
##    yfit = gauss(freq,coeffs[0],coeffs[1],coeffs[2],\
##               coeffs[3],coeffs[4],coeffs[5])
##    zfArray[0] = coeffs[2] # D
##    zfArray[1] = coeffs[3] # E
##    zfArray[2] = coeffs[4] # width
##    zfArray[3] = coeffs[5] # offset
##    zfArray[4] = yStdErr[2] # std err of D
##    zfArray[5] = yStdErr[3] # std err of E
##    zfArray[6] = yStdErr[4] # std err of width
##    zfArray[7] = yStdErr[5] # std err of offset
##    plt.plot(freq,signal,'r-',freq,yguess,'g-',freq,yfit,'b-')
##    plt.show()
##    return zfArray
######################################
######################################
# End fitting function
######################################
######################################


######################################
# NV or Co?
######################################
sample = '200mv'
startIndex = 1
stopIndex = 42
######################################
# Sum data sets and save as separate file
##for j in range(100,1050,50):
##    freq, signal = np.loadtxt('%smv-%s.txt'%(j,startIndex),\
##                              delimiter=', ',skiprows=1,\
##                              unpack=True)
##    for i in range((startIndex+1),stopIndex,1):
##        freq, signaltmp = np.loadtxt('%smv-%s.txt'%(j,i)\
##                              ,delimiter=', ',skiprows=1,\
##                              unpack=True)
##        signal = signal + signaltmp
#################################
##x = 0
##for j in range(450,500,1):
##    x = x + signal[j]
##x = x/50
####print x
##for k in range(0,120,1):
##    signal[k] = x
##for k in range (500,560,1):
##    signal[k] = x
#################################
##    NVsumArray = np.column_stack([freq,signal])
##    np.savetxt('%smv-sum.txt'%j,NVsumArray,delimiter=', ',\
##               header='freq, data', comments='')

### Analyze data sum for D, E, width, and offset with standar errors
##freq,signal = np.loadtxt('%s-sum.txt'%sample,delimiter=', ',skiprows=1,\
##                          unpack=True)
##zfArray = np.array([2.871e9,3.5e6,5e6,35.5,\
##                0,0,0,0])
##zfArray = de('%s-sum'%sample,zfArray)
##print zfArray
##freq,signal = np.loadtxt('600mv-sum.txt',delimiter=', ',skiprows=1,\
##                          unpack=True)
##plt.plot(freq,signal)
##freq1,signal1 = np.loadtxt('Gd-274K-2.txt',delimiter=', ',skiprows=1,\
##                          unpack=True)
##signal = signal/np.amax(signal)
##signal1 = signal1/np.amax(signal1)
##plt.plot(freq,signal,'b-',freq1,signal1,'r-')
##plt.show()

######################################
######################################
# Calculation of B from NV/Co fit
######################################
######################################
##D,E,width,offset,seD,seE,sewidth,seoffset =\
##    np.loadtxt('294K_vs_248K.txt',delimiter=', ',\
##               skiprows=1,unpack=True)
### index 0 is NV, index 1 is Co
##gamma = 28024951642  # G.R. in Hz/T
##H = (E[1]-E[0])/(2*gamma)
### Adjust the uncertainty in ECo to account for the change in width
##seE[1] = seE[1] + ((width[1]-width[0])/2)
####print seE[1]
### Propagate the standard errors from ECo and ENV to H
##seH = H*((np.sqrt(seE[0])**2)+(np.sqrt(seE[1])**2))/(E[1]-E[0])
##print H
##print seH









##zfArray = np.array([2.871e9,3.5e6,3.5e6,0,\
##                    0,0,0,0])
##zfArray = de('Co-23',zfArray)
##print zfArray


##print 'index, D, E, width, offset, seD, seE, sewidth,seoffset,amax'
##for i in range(14,40,2):
##    freq,signal = np.loadtxt('na-%s.txt'%i,delimiter=', ',\
##                             skiprows=1,unpack=True)
##    zfArray = np.array([2.871e9,3.5e6,3.5e6,0,\
##                    0,0,0,0])
##    zfArray = de('na-%s'%i,zfArray)
##    print str(i) + ', ' +\
##          str(zfArray[0]) + ', ' +\
##          str(zfArray[1]) + ', ' +\
##          str(zfArray[2]) + ', ' +\
##          str(zfArray[3]) + ', ' +\
##          str(zfArray[4]) + ', ' +\
##          str(zfArray[5]) + ', ' +\
##          str(zfArray[6]) + ', ' +\
##          str(zfArray[7]) + ', ' +\
##          str(np.amax(signal))


###########################
# Plotting
###########################
##index,D,E,width,offset,seD,seE,sewidth,seoffset,amax = \
##    np.loadtxt('data-Co.txt',delimiter=', ',skiprows=1,unpack=True)
##index1,D1,E1,width1,offset1,seD1,seE1,sewidth1,seoffset1,amax1 = \
##    np.loadtxt('data-na.txt',delimiter=', ',skiprows=1,unpack=True)
##f, axarr = plt.subplots(2,2)
##axarr[0,0].plot(index,D,'b-',index1,D1,'r-')
##axarr[0,0].errorbar(index,D,seD)
##axarr[0,0].errorbar(index1,D1,seD1)
##axarr[0,0].set_title('D')
##axarr[1,0].plot(index,E,'b-',index1,E1,'r-')
##axarr[1,0].errorbar(index,E,seE)
##axarr[1,0].errorbar(index1,E1,seE1)
##axarr[1,0].set_title('E')
##axarr[0,1].plot(index,width,'b-',index1,width1,'r-')
##axarr[0,1].errorbar(index,width,sewidth)
##axarr[0,1].errorbar(index1,width1,sewidth1)
##axarr[0,1].set_title('width')
##axarr[1,1].plot(index,offset,'b-',index1,offset1,'r-')
##axarr[1,1].errorbar(index,offset,seoffset)
##axarr[1,1].errorbar(index1,offset1,seoffset1)
##axarr[1,1].set_title('offset')
##f.subplots_adjust(hspace=.5,wspace=.5)
##plt.show()
