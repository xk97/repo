# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 20:06:32 2018

@author: xk97 
"""
import numpy as np
hour  =  ["%02d:00"  %  i  for  i  in  range(0,  24,  3)]
day  =  ["Mon",  "Tue",  "Wed",  "Thu",  "Fri",  "Sat",  "Sun"]
features  =    day  +  hour

x = list(range(10))
print(x)
y = [x, x]
x = np.power(x, 2)
x = []

#%%

import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

#    if x.ndim != 1:
#        raise ValueError, "smooth only accepts 1 dimension arrays."
#
#    if x.size < window_len:
#        raise ValueError, "Input vector needs to be bigger than window size."
#
#
#    if window_len<3:
#        return x


#    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y




import matplotlib.pylab as plt

def smooth_demo():

    t=np.linspace(-4,4,100)
    x=np.sin(t)
    xn=x+np.random.randn(len(t))*0.1
    y=smooth(x)

    ws=31

    plt.subplot(211)
    plt.plot(np.ones(ws))

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    # plt.hold(True)
    for w in windows[1:]:
        eval('plt.plot('+w+'(ws) )')

    plt.axis([0,30,0,1.1])

    plt.legend(windows)
    plt.title("The smoothing windows")
    plt.subplot(212)
    plt.plot(x)
    plt.plot(xn)
    for w in windows:
        plt.plot(smooth(xn,10,w))
    l=['original signal', 'signal with noise']
    l.extend(windows)

    plt.legend(l)
    plt.title("Smoothing a noisy signal")
    plt.show()

def smooth2(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__=='__main__':
    # smooth_demo()
    
    x = np.linspace(0,2*np.pi,100)
    y = np.sin(x) + np.random.random(100) * 0.8

    plt.plot(x, y,'o')
    plt.plot(x, smooth2(y,3), 'r-', lw=2)
    plt.plot(x, smooth2(y,19), 'g-', lw=2)

    from scipy import signal
    sig = np.repeat([0., 1., 0.], 100)
    win = signal.windows.hann(50)
    filtered = signal.convolve(sig, win, mode='same') / sum(win)
    plt.plot(win)
    plt.plot(sig)
    plt.plot(filtered)
    plt.show()


















