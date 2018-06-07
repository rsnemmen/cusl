/* 
  REATTACH THIS LATER TO bessel.cuh
*/



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