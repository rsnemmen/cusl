"""
Plots the modified Bessel function of the second kind
"""

import pylab, numpy

x,y = numpy.loadtxt('bessel.dat',unpack=True,usecols=(0,1))
xm,ym = numpy.loadtxt('bessel-mathematica.dat',unpack=True,usecols=(0,1))
xc,yc = numpy.loadtxt('bessel-cuda.dat',unpack=True,usecols=(0,1))

pylab.clf()
pylab.plot(x,y,label="GSL")
pylab.plot(xm,ym, label="Mathematica")
pylab.plot(xc,yc, label="CUDA")
pylab.plot(xc,numpy.abs(yc), label="|CUDA|")
pylab.legend()
pylab.show()
