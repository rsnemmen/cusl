/* These routines compute different Bessel functions.  

   Currently the CUDA implementation only supports order 2<=n<=5.

   Adapted from https://github.com/ampl/gsl/blob/48fbd40c7c9c24913a68251d23bdbd0637bbda20/specfunc/bessel_Kn.c
*/

#include "gsl_machine.h"


/* evaluate a function discarding the status value in a modifiable way */

// #define EVAL_RESULT(fn) \
//    gsl_sf_result result; \
//    int status = fn; \
//    if (status != GSL_SUCCESS) { \
//      GSL_ERROR_VAL(#fn, status, result.val); \
//    } ; \
// return result.val;


/* Coefficients for polynomials that will be solved below
   =======================================================
*/

/*
 Minimax rational approximation for [0,1), peak relative error = 2.04*GSL_DBL_EPSILON.
 Source: http://www.advanpix.com/?p=3812
*/
__constant__ 
static double k0_poly[8] = {
   1.1593151565841244842077226e-01,
   2.7898287891460317300886539e-01,
   2.5248929932161220559969776e-02,
   8.4603509072136578707676406e-04,
   1.4914719243067801775856150e-05,
   1.6271068931224552553548933e-07,
   1.2082660336282566759313543e-09,
   6.6117104672254184399933971e-12
};

__constant__ 
static double i0_poly[7] = {
   1.0000000000000000044974165e+00,
   2.4999999999999822316775454e-01,
   2.7777777777892149148858521e-02,
   1.7361111083544590676709592e-03,
   6.9444476047072424198677755e-05,
   1.9288265756466775034067979e-06,
   3.9908220583262192851839992e-08
};

/*
 Chebyshev expansion for [1,8], peak relative error = 1.28*GSL_DBL_EPSILON. 
 Source: Pavel Holoborodko.
*/
__constant__ 
static double ak0_data[24] = {
  -3.28737867094650101e-02,
  -4.49369057710236880e-02,
  +2.98149992004308095e-03,
  -3.03693649396187920e-04,
  +3.91085569307646836e-05,
  -5.86872422399215952e-06,
  +9.82873709937322009e-07,
  -1.78978645055651171e-07,
  +3.48332306845240957e-08,
  -7.15909210462546599e-09,
  +1.54019930048919494e-09,
  -3.44555485579194210e-10,
  +7.97356101783753023e-11,
  -1.90090968913069735e-11,
  +4.65295609304114621e-12,
  -1.16614287433470780e-12,
  +2.98554375218596891e-13,
  -7.79276979512292169e-14,
  +2.07027467168948402e-14,
  -5.58987860393825313e-15,
  +1.53202965950646914e-15,
  -4.25737536712188186e-16,
  +1.19840238501357389e-16,
  -3.41407346762502397e-17
};

__constant__ 
static cheb_series ak0_cs = {
  ak0_data,
  23,
  -1, 1,
  10
};

/* 
 Chebyshev expansion for [8,inf), peak relative error = 1.25*GSL_DBL_EPSILON.
 Source: SLATEC/dbsk0e.f
*/
__constant__ 
static double ak02_data[14] = {
  -.1201869826307592240E-1,
  -.9174852691025695311E-2,
  +.1444550931775005821E-3,
  -.4013614175435709729E-5,
  +.1567831810852310673E-6,
  -.7770110438521737710E-8,
  +.4611182576179717883E-9,
  -.3158592997860565771E-10,
  +.2435018039365041128E-11,
  -.2074331387398347898E-12,
  +.1925787280589917085E-13,
  -.1927554805838956104E-14,
  +.2062198029197818278E-15,
  -.2341685117579242403E-16
};



__constant__ 
static cheb_series ak02_cs = {
  ak02_data,
  13,
  -1, 1,
  8
};




/* 
  These routines compute the scaled irregular modified cylindrical 
  Bessel function of zeroth order \exp(x) K_0(x) for x>0. 
*/
__device__
double cu_sf_bessel_K0_scaled_e(const double x)
{
  /* CHECK_POINTER(result) */
  double c, result;

  if(x <= 0.0) {
    //DOMAIN_ERROR(result);
    printf("error! x<=0\n");
    return -1E6;
  }
  else if(x < 1.0) {
    const double lx = log(x);
    const double ex = exp(x);
    const double x2 = x*x;
    result = ex * (cu_poly_eval(k0_poly,8,x2)-lx*(1.0+0.25*x2*cu_poly_eval(i0_poly,7,0.25*x2)));
  }
  else if(x <= 8.0) {
    const double sx = sqrt(x);
    c=cheb_eval_e(&ak0_cs, (16.0/x-9.0)/7.0);
    result = (1.203125 + c) / sx; /* 1.203125 = 77/64 */
  }
  else {
    const double sx = sqrt(x);
    c=cheb_eval_e(&ak02_cs, 16.0/x-1.0);
    result= (1.25 + c) / sx;
  } 

  return result;
}

