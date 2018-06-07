"""
Compares the plots of GSL Bessel functions with the CUSL 
Bessel functions.
"""

import pylab, numpy

x,k0,k1,kn = numpy.loadtxt('bessel-gsl.dat',unpack=True,usecols=(0,1,2,3))
xc,k0c,k1c,knc = numpy.loadtxt('bessel-cuda-cusl.dat',unpack=True,usecols=(0,1,2,4))

pylab.clf()

pylab.plot(x,k0,label="$K_0$, GSL")
pylab.plot(x,k1,label="$K_1$, GSL")
pylab.plot(x,kn,label="$K_n$, GSL")


pylab.plot(xc,k0c, '.', label="$K_0$, CUDA")
pylab.plot(xc,k1c, '.', label="$K_1$, CUDA")
pylab.plot(xc,knc, '.', label="$K_n$, CUDA")

pylab.ylim(0,10)
pylab.legend()
pylab.show()
