"""
Compares the plots of GSL Bessel functions with the CUSL 
Bessel functions.
"""

import pylab, numpy

x,k0,k1,kn = numpy.loadtxt('bessel-gsl.dat',unpack=True,usecols=(0,1,2,3))
xc,k0c,i0,i1 = numpy.loadtxt('bessel-cuda-cusl.dat',unpack=True,usecols=(0,1,2,3))

pylab.clf()

#pylab.plot(x,kn,label="$K_n$, GSL")
pylab.plot(x,k0,label="$K_0$, GSL")
#pylab.plot(x,k1,label="$K_1$, GSL")

pylab.plot(xc,k0c, label="$K_0$, CUDA")
#pylab.plot(xc,i0, 'o', label="$I_0$, CUDA")
#pylab.plot(xc,i1, 'o', label="$I_1$, CUDA")

#pylab.ylim(0,10)
pylab.legend()
pylab.show()
