/* The Gamma distribution of order a>0 is defined by:
   p(x) dx = {1 / \Gamma(a) b^a } x^{a-1} e^{-x/b} dx
   for x>0.  If X and Y are independent gamma-distributed random
   variables of order a1 and a2 with the same scale parameter b, then
   X+Y has gamma distribution of order a1+a2.
   The algorithms below are from Knuth, vol 2, 2nd ed, p. 129. 
*/

double
gsl_ran_gamma (const gsl_rng * r, const double a, const double b)
{
  /* assume a > 0 */

  if (a < 1)
    {
      double u = gsl_rng_uniform_pos (r);
      return gsl_ran_gamma (r, 1.0 + a, b) * pow (u, 1.0 / a);
    }

  {
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1)
      {
        do
          {
            x = gsl_ran_gaussian_ziggurat (r, 1.0);
            v = 1.0 + c * x;
          }
        while (v <= 0);

        v = v * v * v;
        u = gsl_rng_uniform_pos (r);

        if (u < 1 - 0.0331 * x * x * x * x) 
          break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
          break;
      }
    
    return b * d * v;
  }
}