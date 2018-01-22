# The "esr" function controls various instruments 
# (HP8665A frequency generator, SR510 lock-in amplifier, Keithley digital voltmeter) 
# to perform electron spin resonance experiments on nitrogen-vacancy centers

import visa
import time
from datetime import datetime
import serial
import os
import numpy as np
import matplotlib.pyplot as plt

rm = visa.ResourceManager()
rm.list_resources()

def esr(startFreq, stopFreq, stepFreq, delay, counts, filename):      
    HP8665A = rm.open_resource('GPIB0::11::INSTR')
    keithley = rm.open_resource('GPIB0::5::INSTR')
    index = 0
    dataArray = np.zeros([((int(stopFreq)-int(startFreq))/int(stepFreq)), 2], dtype=float)

    plt.ion()
    fig=plt.figure()
    plt.xlim(startFreq,stopFreq)
    
    try:
        for freq in range(int(startFreq), int(stopFreq), int(stepFreq)):
            HP8665A.write('freq %shz' % freq)
            time.sleep(delay)
        
            i = 0
            asum = 0
            queryArray = np.zeros(counts, dtype=float)
            while i < counts:
                a = keithley.query('')
                aRelevant = a[4:]
                asum = asum + float(aRelevant)
                i = i+1
            dataArray[index,0] = freq
            dataArray[index,1] = asum
            if index != 0:
                lines.remove()
            lines, = plt.plot(dataArray[:index+1,0],dataArray[:index+1,1],'bo-')
            plt.show()
            plt.pause(0.0001)
            index = index + 1

    except KeyboardInterrupt:
        np.savetxt('%s-interrupted.txt' % filename, dataArray, delimiter=', ', header='freq, data', comments='')
        plt.ioff()
        plt.close()
        os._exit(0)
    #print dataArray
    plt.ioff()
    plt.close()
    np.savetxt('%s.txt' % filename, dataArray, delimiter=', ', header='%s_freq, %s_data'%(filename,filename), comments='')
    HP8665A.write('freq %shz' % startFreq)
    time.sleep(1)
    #x, y = np.loadtxt('%s.txt' % filename, delimiter=', ', skiprows=1, unpack=True)
    #plt.plot(x,y, label='')
    #plt.show()


# This code is modified to run various forms of data collection
HP8665A = rm.open_resource('GPIB0::11::INSTR')
for i in range(1,10000,1):
    for amp in range(100,1000,50):
        HP8665A.write('ampl %smv' % amp)
        esr(2.8e9,2.94e9,.5e6,.2,10,'%smv-%s'%(amp,i))
        print datetime.now()
##freq,signal = np.loadtxt('Co-20.txt',delimiter=', ',skiprows=1,unpack=True)
##plt.plot(freq,signal,'b-')
##plt.show()
