/* These routines compute the irregular modified cylindrical Bessel 
   function of order n, K_n(x), for x > 0. 

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
  result.val (GSL) => result (CUSL)
*/
__device__
double cu_sf_bessel_Kn(const int n, const double x)
{
	//EVAL_RESULT(gsl_sf_bessel_Kn_e(n, x, &result));
	//const int status = gsl_sf_bessel_Kn_scaled_e(n, x, result);

	n = abs(n); /* K(-n, z) = K(n, z) */

	/* CHECK_POINTER(result) */

	if(x <= 0.0) {
		printf("Invalid x for Bessel\n");
	  	return; 
	}
	else if(n == 0) {
		printf("n=0 not yet supported in CUDA Bessel\n");
	  	//return gsl_sf_bessel_K0_scaled_e(x, result);
	  	return;
	}
	else if(n == 1) {
		printf("n=1 not yet supported in CUDA Bessel\n");	
	  	//return gsl_sf_bessel_K1_scaled_e(x, result);
	  	return;
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

	/*
	===============================================
	*/

	const double ex = exp(-x);
	result *= ex;

	return result;
}