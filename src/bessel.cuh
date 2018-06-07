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


/* Coefficients for polynomials that will be used below
   =====================================================
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
 Minimax rational approximation for [0,1), peak relative error = 1.83*GSL_DBL_EPSILON.
 Source: http://www.advanpix.com/?p=3987
*/
__constant__ 
static double k1_poly[9] = {
  -3.0796575782920622440538935e-01,
  -8.5370719728650778045782736e-02,
  -4.6421827664715603298154971e-03,
  -1.1253607036630425931072996e-04,
  -1.5592887702110907110292728e-06,
  -1.4030163679125934402498239e-08,
  -8.8718998640336832196558868e-11,
  -4.1614323580221539328960335e-13,
  -1.5261293392975541707230366e-15
};

__constant__ 
static double i1_poly[7] = {
  8.3333333333333325191635191e-02,
  6.9444444444467956461838830e-03,
  3.4722222211230452695165215e-04,
  1.1574075952009842696580084e-05,
  2.7555870002088181016676934e-07,
  4.9724386164128529514040614e-09
};

/*
 Chebyshev expansion for [1,8], peak relative error = 1.28*GSL_DBL_EPSILON. 
 Source: Pavel Holoborodko.
*/
__constant__ 
static double ak1_data[25] = {
  +2.07996868001418246e-01,
  +1.62581565017881476e-01,
  -5.87070423518863640e-03,
  +4.95021520115789501e-04,
  -5.78958347598556986e-05,
  +8.18614610209334726e-06,
  -1.31604832009487277e-06,
  +2.32546031520101213e-07,
  -4.42206518311557987e-08,
  +8.92163994883100361e-09,
  -1.89046270526983427e-09,
  +4.17568808108504702e-10,
  -9.55912361791375794e-11,
  +2.25769353153867758e-11,
  -5.48128000211158482e-12,
  +1.36386122546441926e-12,
  -3.46936690565986409e-13,
  +9.00354564415705942e-14,
  -2.37950577776254432e-14,
  +6.39447503964025336e-15,
  -1.74498363492322044e-15,
  +4.82994547989290473e-16,
  -1.35460927805445606e-16,
  +3.84604274446777234e-17,
  -1.10456856122581316e-17
};

__constant__ 
static cheb_series ak1_cs = {
  ak1_data,
  24,
  -1, 1,
  9
};

/* 
 Chebyshev expansion for [8,inf), peak relative error = 1.25*GSL_DBL_EPSILON.
 Source: SLATEC/dbsk1e.f
*/
__constant__ 
static double ak12_data[14] = {
  +.637930834373900104E-1,
  +.283288781304972094E-1,
  -.247537067390525035E-3,
  +.577197245160724882E-5,
  -.206893921953654830E-6,
  +.973998344138180418E-8,
  -.558533614038062498E-9,
  +.373299663404618524E-10,
  -.282505196102322545E-11,
  +.237201900248414417E-12,
  -.217667738799175398E-13,
  +.215791416161603245E-14,
  -.229019693071826928E-15,
  +.258288572982327496E-16
};

__constant__ 
static cheb_series ak12_cs = {
  ak12_data,
  13,
  -1, 1,
  7
};






/*
  Function definitions
  =====================
*/



/* Debye functions [Abramowitz+Stegun, 9.3.9-10] */
__device__
inline static double 
debye_u1(const double * tpow)
{
  return (3.0*tpow[1] - 5.0*tpow[3])/24.0;
}

__device__
inline static double 
debye_u2(const double * tpow)
{
  return (81.0*tpow[2] - 462.0*tpow[4] + 385.0*tpow[6])/1152.0;
}

__device__
inline
static double debye_u3(const double * tpow)
{
  return (30375.0*tpow[3] - 369603.0*tpow[5] + 765765.0*tpow[7] - 425425.0*tpow[9])/414720.0;
}

__device__
inline
static double debye_u4(const double * tpow)
{
  return (4465125.0*tpow[4] - 94121676.0*tpow[6] + 349922430.0*tpow[8] - 
          446185740.0*tpow[10] + 185910725.0*tpow[12])/39813120.0;
}

__device__
inline
static double debye_u5(const double * tpow)
{
  return (1519035525.0*tpow[5]     - 49286948607.0*tpow[7] + 
          284499769554.0*tpow[9]   - 614135872350.0*tpow[11] + 
          566098157625.0*tpow[13]  - 188699385875.0*tpow[15])/6688604160.0;
}

#if 0
inline
static double debye_u6(const double * tpow)
{
  return (2757049477875.0*tpow[6] - 127577298354750.0*tpow[8] + 
          1050760774457901.0*tpow[10] - 3369032068261860.0*tpow[12] + 
          5104696716244125.0*tpow[14] - 3685299006138750.0*tpow[16] + 
          1023694168371875.0*tpow[18])/4815794995200.0;
}
#endif




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






__device__
double cu_sf_bessel_K1_scaled_e(const double x)
{
  /* CHECK_POINTER(result) */
  double c;

  if(x <= 0.0) {
    //DOMAIN_ERROR(result);
    printf("error, x<=0");
    return -1E6;
  }
  else if(x < 2.0*GSL_DBL_MIN) {
    printf("error, overflow");
    return -1E6;
  }
  else if(x < 1.0) {
    const double lx = log(x);
    const double ex = exp(x);
    const double x2 = x*x;
    const double t  = 0.25*x2;    
    const double i1 = 0.5 * x * (1.0 + t * (0.5 + t * cu_poly_eval(i1_poly,6,t)));
    return ex * (x2 * cu_poly_eval(k1_poly,9,x2) + x * lx * i1 + 1) / x;
  }
  else if(x <= 8.0) {
    const double sx = sqrt(x);
    c=cheb_eval_e(&ak1_cs, (16.0/x-9.0)/7.0);
    return (1.375 + c) / sx; /* 1.375 = 11/8 */
  }
  else {
    const double sx = sqrt(x);
    c=cheb_eval_e(&ak12_cs, 16.0/x-1.0);
    return (1.25 + c) / sx;
  }
}






/*
   [Abramowitz+Stegun, 9.6.11]
   assumes n >= 1
*/
__device__ 
static double bessel_Kn_scaled_small_x(const int n, const double x)
{
  int k;
  double y = 0.25 * x * x;
  double ln_x_2 = log(0.5*x);
  double ex = exp(x);
  double ln_nm1_fact;
  double k_term;
  double term1, sum1, ln_pre1;
  double term2, sum2, pre2;

  ln_nm1_fact=cu_sf_lnfact_e((unsigned int)(n-1));

  ln_pre1 = -n*ln_x_2 + ln_nm1_fact;
  if(ln_pre1 > GSL_LOG_DBL_MAX - 3.0) {
    printf("error: overflow\n");
    return 1E50;
  }

  sum1 = 1.0;
  k_term = 1.0;
  for(k=1; k<=n-1; k++) {
    k_term *= -y/(k * (n-k));
    sum1 += k_term;
  }
  term1 = 0.5 * exp(ln_pre1) * sum1;

  pre2 = 0.5 * exp(n*ln_x_2);
  if(pre2 > 0.0) {
    const int KMAX = 20;
    double psi_n, npk_fact;
    double yk = 1.0;
    double k_fact  = 1.0;
    double psi_kp1 = -M_EULER;
    double psi_npkp1;
    psi_n=cu_sf_psi_int_e(n);
    npk_fact= cu_sf_fact_e((unsigned int)n);
    psi_npkp1 = psi_n + 1.0/n;
    sum2 = (psi_kp1 + psi_npkp1 - 2.0*ln_x_2)/npk_fact;
    for(k=1; k<KMAX; k++) {
      psi_kp1   += 1.0/k;
      psi_npkp1 += 1.0/(n+k);
      k_fact    *= k;
      npk_fact *= n+k;
      yk *= y;
      k_term = yk*(psi_kp1 + psi_npkp1 - 2.0*ln_x_2)/(k_fact*npk_fact);
      sum2 += k_term;
    }
    term2 = ( IS_ODD(n) ? -1.0 : 1.0 ) * pre2 * sum2;
  }
  else {
    term2 = 0.0;
  }

  return ex * (term1 + term2);
}




/* 

  x >> nu*nu+1
*/
__device__
double cu_sf_bessel_Knu_scaled_asympx_e(const double nu, const double x)
{
  double mu   = 4.0*nu*nu;
  double mum1 = mu-1.0;
  double mum9 = mu-9.0;
  double pre  = sqrt(M_PI/(2.0*x));
  double r    = nu/x;
  return pre * (1.0 + mum1/(8.0*x) + mum1*mum9/(128.0*x*x));
}




/* nu -> Inf; uniform in x > 0  [Abramowitz+Stegun, 9.7.8]
 *
 * error:
 *   identical to that above for Inu_scaled
 */
__device__
double cu_sf_bessel_Knu_scaled_asymp_unif_e(const double nu, const double x)
{
  int i;
  double z = x/nu;
  double root_term = hypot(1.0,z);
  double pre = sqrt(M_PI/(2.0*nu*root_term));
  double eta = root_term + log(z/(1.0+root_term));
  double ex_arg = ( z < 1.0/GSL_ROOT3_DBL_EPSILON ? nu*(z - eta) : 0.5*nu/z*(1.0 + 1.0/(12.0*z*z)) );
  double ex_result = exp(ex_arg);

  double t = 1.0/root_term;
  double sum;
  double tpow[16];
  tpow[0] = 1.0;
  for(i=1; i<16; i++) tpow[i] = t * tpow[i-1];
  sum = 1.0 - debye_u1(tpow)/nu + debye_u2(tpow)/(nu*nu) - debye_u3(tpow)/(nu*nu*nu)
        + debye_u4(tpow)/(nu*nu*nu*nu) - debye_u5(tpow)/(nu*nu*nu*nu*nu);

  return pre * ex_result * sum;
}
