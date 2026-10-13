"""Q08 synthetic scaling-law fit and regime-shift warning. Run: python3 regime_shift.py"""
import math

def fit_power(xs, ys):
    # log y = intercept + slope*log x, least squares without external libraries
    lx=[math.log(x) for x in xs]; ly=[math.log(y) for y in ys]; n=len(xs)
    slope=(n*sum(a*b for a,b in zip(lx,ly))-sum(lx)*sum(ly))/(n*sum(a*a for a in lx)-sum(lx)**2)
    intercept=(sum(ly)-slope*sum(lx))/n
    return math.exp(intercept), slope

def check():
    xs=[1,2,4,8,16]; ys=[10*x**-0.5 for x in xs]
    a,b=fit_power(xs,ys); interp=a*32**b
    true=10*32**-0.5
    assert abs(interp-true)/true < 1e-9
    shifted=3.0 # invented new regime at x=64; not governed by prior curve
    extrap=a*64**b
    assert abs(extrap-shifted) > 1.0
    print(f'PASS: stable interpolation={interp:.3f} true={true:.3f}; regime-shift extrapolation={extrap:.3f} actual={shifted:.3f}')
if __name__=='__main__': check()
