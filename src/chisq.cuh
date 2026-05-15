/* The chisq distribution has the form
   p(x) dx = (1/(2*Gamma(nu/2))) (x/2)^(nu/2 - 1) exp(-x/2) dx
   for x = 0 ... +infty 

   Code taken from https://github.com/ampl/gsl/blob/48fbd40c7c9c24913a68251d23bdbd0637bbda20/randist/chisq.c
*/

#ifndef CUSL_NAN
#define CUSL_NAN (0.0/0.0)
#endif

__device__
double cu_ran_chisq_double(curandState *r, const double nu)
{
  if (nu <= 0.0) {
    return CUSL_NAN;
  }

  double chisq = 2 * cu_ran_gamma_double(r, nu / 2, 1.0);
  return chisq;
}
