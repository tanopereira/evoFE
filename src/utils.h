#ifndef EVOFE_UTILS_H
#define EVOFE_UTILS_H

#include <cmath>
#include <algorithm>

namespace evofe {

// 1D Brent minimization on interval [a, b]
template <typename Func>
inline double brent_minimize(Func&& f, double a, double b, double tol = 1e-5, int max_iter = 100) {
  const double golden = 0.5 * (3.0 - std::sqrt(5.0)); // 0.381966011250105
  double x = a + golden * (b - a);
  double w = x;
  double v = w;
  double d = 0.0;
  double e = 0.0;
  double fx = f(x);
  double fw = fx;
  double fv = fw;

  for (int iter = 0; iter < max_iter; ++iter) {
    double m = 0.5 * (a + b);
    double tol1 = tol * std::abs(x) + 1e-10;
    double tol2 = 2.0 * tol1;

    if (std::abs(x - m) <= tol2 - 0.5 * (b - a)) {
      break;
    }

    double p = 0.0, q = 0.0, r = 0.0;
    if (std::abs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = std::abs(q);
      double etemp = e;
      e = d;
      if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
        e = (x >= m) ? a - x : b - x;
        d = golden * e;
      } else {
        d = p / q;
        double u = x + d;
        if (u - a < tol2 || b - u < tol2) {
          d = (m - x >= 0.0) ? tol1 : -tol1;
        }
      }
    } else {
      e = (x >= m) ? a - x : b - x;
      d = golden * e;
    }

    double u = (std::abs(d) >= tol1) ? x + d : x + ((d >= 0.0) ? tol1 : -tol1);
    double fu = f(u);

    if (fu <= fx) {
      if (u >= x) a = x; else b = x;
      v = w; fv = fw;
      w = x; fw = fx;
      x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
    }
  }
  return x;
}

} // namespace evofe

#endif // EVOFE_UTILS_H
