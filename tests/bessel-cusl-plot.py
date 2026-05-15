"""
Compares the plots of GSL Bessel functions with the CUSL 
Bessel functions.
"""

import pylab, numpy

x,k0,k1,k2,k3,k4,k5 = numpy.loadtxt('bessel-gsl.dat',unpack=True,usecols=(0,1,2,3,4,5,6))
xc,k0c,k1c,k2c,k3c,k4c,k5c = numpy.loadtxt('bessel-cuda-cusl.dat',unpack=True,usecols=(0,1,2,3,4,5,6))

pylab.clf()

pylab.plot(x,k0,label="$K_0$, GSL")
pylab.plot(x,k1,label="$K_1$, GSL")
pylab.plot(x,k2,label="$K_2$, GSL")
pylab.plot(x,k3,label="$K_3$, GSL")
pylab.plot(x,k4,label="$K_4$, GSL")
pylab.plot(x,k5,label="$K_5$, GSL")


pylab.plot(xc,k0c, '.', label="$K_0$, CUDA")
pylab.plot(xc,k1c, '.', label="$K_1$, CUDA")
pylab.plot(xc,k2c, '.', label="$K_2$, CUDA")
pylab.plot(xc,k3c, '.', label="$K_3$, CUDA")
pylab.plot(xc,k4c, '.', label="$K_4$, CUDA")
pylab.plot(xc,k5c, '.', label="$K_5$, CUDA")

pylab.ylim(0,10)
pylab.legend()
pylab.show()
