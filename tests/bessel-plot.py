"""
Compares the plots of GSL modified Bessel function of the second kind
with the CUDA Math Library Bessel functions.
"""

import pylab, numpy

x,k0,k1,kn = numpy.loadtxt('bessel-gsl.dat',unpack=True,usecols=(0,1,2,3))
xm,ym = numpy.loadtxt('bessel-mathematica.dat',unpack=True,usecols=(0,1))
xc,yn,i0,i1 = numpy.loadtxt('bessel-cuda.dat',unpack=True,usecols=(0,1,2,3))

pylab.clf()

pylab.plot(x,kn,label="$K_n$, GSL")
pylab.plot(x,k0,label="$K_0$, GSL")
pylab.plot(x,k1,label="$K_1$, GSL")

pylab.plot(xm,ym, label="Mathematica")

pylab.plot(xc,yn, 'o', label="$Y_n$, CUDA")
pylab.plot(xc,i0, 'o', label="$I_0$, CUDA")
pylab.plot(xc,i1, 'o', label="$I_1$, CUDA")

pylab.ylim(0,10)
pylab.legend()
pylab.show()
