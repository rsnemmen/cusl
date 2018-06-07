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
  double c;

  if(x <= 0.0) {
    //DOMAIN_ERROR(result);
    printf("error! x<=0\n");
    return;
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



/*
   [Abramowitz+Stegun, 9.6.11]
   assumes n >= 1
*/
__device__ static
int bessel_Kn_scaled_small_x(const int n, const double x, result)
{
  int k;
  double y = 0.25 * x * x;
  double ln_x_2 = log(0.5*x);
  double ex = exp(x);
  double ln_nm1_fact;
  double k_term;
  double term1, sum1, ln_pre1;
  double term2, sum2, pre2;

  gsl_sf_lnfact_e((unsigned int)(n-1), &ln_nm1_fact);

  ln_pre1 = -n*ln_x_2 + ln_nm1_fact.val;
  if(ln_pre1 > GSL_LOG_DBL_MAX - 3.0) GSL_ERROR ("error", GSL_EOVRFLW);

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
    gsl_sf_result psi_n;
    gsl_sf_result npk_fact;
    double yk = 1.0;
    double k_fact  = 1.0;
    double psi_kp1 = -M_EULER;
    double psi_npkp1;
    gsl_sf_psi_int_e(n, &psi_n);
    gsl_sf_fact_e((unsigned int)n, &npk_fact);
    psi_npkp1 = psi_n.val + 1.0/n;
    sum2 = (psi_kp1 + psi_npkp1 - 2.0*ln_x_2)/npk_fact.val;
    for(k=1; k<KMAX; k++) {
      psi_kp1   += 1.0/k;
      psi_npkp1 += 1.0/(n+k);
      k_fact    *= k;
      npk_fact.val *= n+k;
      yk *= y;
      k_term = yk*(psi_kp1 + psi_npkp1 - 2.0*ln_x_2)/(k_fact*npk_fact.val);
      sum2 += k_term;
    }
    term2 = ( GSL_IS_ODD(n) ? -1.0 : 1.0 ) * pre2 * sum2;
  }
  else {
    term2 = 0.0;
  }

  result->val  = ex * (term1 + term2);
  result->err  = ex * GSL_DBL_EPSILON * (fabs(ln_pre1)*fabs(term1) + fabs(term2));
  result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);

  return GSL_SUCCESS;
}





__device__
int gsl_sf_bessel_Kn_scaled_e(int n, const double x, gsl_sf_result * result)
{
  n = abs(n); /* K(-n, z) = K(n, z) */

  /* CHECK_POINTER(result) */

  if(x <= 0.0) {
    DOMAIN_ERROR(result);
  }
  else if(n == 0) {
    return gsl_sf_bessel_K0_scaled_e(x, result);
  }
  else if(n == 1) {
    return gsl_sf_bessel_K1_scaled_e(x, result);
  }
  else if(x <= 5.0) {
    return bessel_Kn_scaled_small_x(n, x, result);
  }
  else if(GSL_ROOT3_DBL_EPSILON * x > 0.25 * (n*n + 1)) {
    return gsl_sf_bessel_Knu_scaled_asympx_e((double)n, x, result);
  }
  else if(GSL_MIN(0.29/(n*n), 0.5/(n*n + x*x)) < GSL_ROOT3_DBL_EPSILON) {
    return gsl_sf_bessel_Knu_scaled_asymp_unif_e((double)n, x, result);
  }
  else {
    /* Upward recurrence. [Gradshteyn + Ryzhik, 8.471.1] */
    double two_over_x = 2.0/x;
    gsl_sf_result r_b_jm1;
    gsl_sf_result r_b_j;
    int stat_0 = gsl_sf_bessel_K0_scaled_e(x, &r_b_jm1);
    int stat_1 = gsl_sf_bessel_K1_scaled_e(x, &r_b_j);
    double b_jm1 = r_b_jm1.val;
    double b_j   = r_b_j.val;
    double b_jp1;
    int j;

    for(j=1; j<n; j++) {
      b_jp1 = b_jm1 + j * two_over_x * b_j;
      b_jm1 = b_j;
      b_j   = b_jp1; 
    } 
    
    result->val  = b_j;
    result->err  = n * (fabs(b_j) * (fabs(r_b_jm1.err/r_b_jm1.val) + fabs(r_b_j.err/r_b_j.val)));
    result->err += 2.0 * GSL_DBL_EPSILON * fabs(result->val);

    return GSL_ERROR_SELECT_2(stat_0, stat_1);
  }
}


// __device__
// int gsl_sf_bessel_Kn_e(const int n, const double x, double result)
// {
//   const double ex = exp(-x);
//   result->val *= ex;
//   result->err *= ex;
//   result->err += x * GSL_DBL_EPSILON * fabs(result->val);
//   return status;
// }


// __device__
// double gsl_sf_bessel_Kn_scaled(const int n, const double x)
// {
//   EVAL_RESULT(gsl_sf_bessel_Kn_scaled_e(n, x, &result));
// }


/*
  These routines compute the irregular modified cylindrical Bessel 
  function of order n, K_n(x), for x > 0. It is also called simply the 
  modified Bessel function of the second kind, function BesselK[2,x] 
  in Mathematica).

  result.val (GSL) => result (CUSL)
*/
__device__
double cu_sf_bessel_Kn(int n, const double x)
{
	double result;

	n = abs(n); /* K(-n, z) = K(n, z) */

	/* CHECK_POINTER(result) */

	if(x <= 0.0) {
		printf("Invalid x for Bessel\n");
	  	return; 
	}
	else if(n == 0) {
		printf("n=0 not yet supported in CUDA Bessel\n");
	  	result=gsl_sf_bessel_K0_scaled_e(x);
	}
	else if(n == 1) {
		printf("n=1 not yet supported in CUDA Bessel\n");	
	  	result=gsl_sf_bessel_K1_scaled_e(x);
	}
	else if(x <= 5.0) {
	  result=bessel_Kn_scaled_small_x(n, x);
	}
	else if(GSL_ROOT3_DBL_EPSILON * x > 0.25 * (n*n + 1)) {
	  result=gsl_sf_bessel_Knu_scaled_asympx_e((double)n, x);
	}
	else if(fmin(0.29/(n*n), 0.5/(n*n + x*x)) < GSL_ROOT3_DBL_EPSILON) {
	  result=gsl_sf_bessel_Knu_scaled_asymp_unif_e((double)n, x);
	}
	else {
	  /* Upward recurrence. [Gradshteyn + Ryzhik, 8.471.1] */
	  double two_over_x = 2.0/x;
	  //gsl_sf_result r_b_jm1;
	  //gsl_sf_result r_b_j;
	  double b_jm1 = gsl_sf_bessel_K0_scaled_e(x);
	  double b_j = gsl_sf_bessel_K1_scaled_e(x);
	  //double b_jm1 = r_b_jm1.val;
	  //double b_j   = r_b_j.val;
	  double b_jp1;
	  int j;

	  for(j=1; j<n; j++) {
	    b_jp1 = b_jm1 + j * two_over_x * b_j;
	    b_jm1 = b_j;
	    b_j   = b_jp1; 
	  } 
	  
	  result = b_j;
	}

	/*
	===============================================
	*/

	result *= exp(-x);

	return result;
}