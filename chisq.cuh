/* The chisq distribution has the form
   p(x) dx = (1/(2*Gamma(nu/2))) (x/2)^(nu/2 - 1) exp(-x/2) dx
   for x = 0 ... +infty 

   Code taken from https://github.com/ampl/gsl/blob/48fbd40c7c9c24913a68251d23bdbd0637bbda20/randist/chisq.c
*/

double
gsl_ran_chisq (const gsl_rng * r, const double nu)
{
  double chisq = 2 * gsl_ran_gamma (r, nu / 2, 1.0);
  return chisq;
}