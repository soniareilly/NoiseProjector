# Noise projector for Poisson data with entropy constraint

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from numpy.polynomial import chebyshev
from scipy.interpolate import CubicSpline
from scipy.sparse import dia_array
import matplotlib.pyplot as plt

os = 2
N = os * 121 + 1
N2 = int(np.floor(N/2) + 1)

# true intensity
R = 0.2
Id = np.array([0.0]*(N*N2))
for i in range(N):
    for j in range(N2):
        x = min(i,N-i)/N2
        y = j/N2
        r = np.sqrt(x**2 + y**2)
        Id[j+i*N2] = np.floor(1000*np.exp(-(r/R)**2/2))
# Plot true intensity
#Idmat = np.reshape(Id,(N,N2))
#plt.imshow(Idmat)
#plt.show()

# current intensity
I = (Id+1e-6)*1.1

# precompute log(k!)/k!
logkk = np.array([0.0]*29)
kbang = 1.0
for k in range(1,30):
    kbang *= k
    logkk[k-1] = np.log(kbang)/kbang

# objective function
def obj(x):
    return sum((x-I)**2)

# objective function derivative wrt x
def obj_der(x):
    return 2*(x-I)

# objective function Hessian
def obj_hess(x):
    return 2*np.identity(x.size)

# entropy (x is scalar)
def H(x):
    if x <= 1:
        sum = 0
        for k in range(1,10):
            sum += x**k * logkk[k-1]
        Hx = x * (1-np.log(x)) + np.exp(-x)*sum
    elif 1 <= x <= 10:
        sum = 0
        for k in range(1,30):
            sum += x**k * logkk[k-1]
        Hx = x * (1-np.log(x)) + np.exp(-x)*sum
    else:
        Hx = 0.5*np.log(2*np.pi*np.e*x) - 1/12/x - 1/24/x**2 - 19/360/x**3
    return Hx

# entropy derivative wrt x
def dH(x):
    if x <= 1:
        sum1 = 0
        sum2 = 0
        for k in range(1,10):
            sum1 += x**k * logkk[k-1]
            sum2 += k*x**(k-1) * logkk[k-1]
        dHx = -np.log(x) - np.exp(-x)*sum1 + np.exp(-x)*sum2
    elif 1 <= x <= 10:
        sum1 = 0
        sum2 = 0
        for k in range(1,30):
            sum1 += x**k * logkk[k-1]
            sum2 += k*x**(k-1) * logkk[k-1]
        dHx = -np.log(x) - np.exp(-x)*sum1 + np.exp(-x)*sum2
    else:
        dHx = 1/2/x + 1/12/x**2 + 1/12/x**3 + 19/120/x**4
    return dHx

# entropy 2nd derivative wrt x
def ddH(x):
    if x <= 1:
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for k in range(1,10):
            sum1 += x**k * logkk[k-1]
            sum2 += k*x**(k-1) * logkk[k-1]
            if k > 1:
                sum3 += k*(k-1)*x**(k-2) * logkk[k-1]
        ddHx = -1/x + np.exp(-x)*sum1 - 2*np.exp(-x)*sum2 + np.exp(-x)*sum3
    elif 1 <= x <= 10:
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for k in range(1,30):
            sum1 += x**k * logkk[k-1]
            sum2 += k*x**(k-1) * logkk[k-1]
            if k > 1:
                sum3 += k*(k-1)*x**(k-2) * logkk[k-1]
        ddHx = -1/x + np.exp(-x)*sum1 - 2*np.exp(-x)*sum2 + np.exp(-x)*sum3
    else:
        ddHx = -1/2/x**2 - 1/6/x**3 - 1/3/x**4 - 19/24/x**5
    return ddHx

# H applied to vector x
def Hvec(x):
    if type(x) == np.ndarray:
        ans = np.zeros_like(x)
        for i in range(len(x)):
            ans[i] = H(x[i])
    else:
        ans = H(x)
    return ans

# dH applied to vector x
def dHvec(x):
    if type(x) == np.ndarray:
        ans = np.zeros_like(x)
        for i in range(len(x)):
            ans[i] = dH(x[i])
    else:
        ans = dH(x)
    return ans

# ddH applied to vector x
def ddHvec(x):
    if type(x) == np.ndarray:
        ans = np.zeros_like(x)
        for i in range(len(x)):
            ans[i] = ddH(x[i])
    else:
        ans = ddH(x)
    return ans

# cubic spline interpolation of H, dH, ddH on [10^-10, 10^4]
# in order to speed up evaluation
logxs = np.linspace(-10.0,4.0,1000)
xs = np.exp(logxs*np.log(10))
Hcubic = CubicSpline(xs,Hvec(xs))
dHcubic = CubicSpline(xs,dHvec(xs))
ddHcubic = CubicSpline(xs,ddHvec(xs))

# constraint function
def cons_f(x):
    f = sum(-Id*np.log(x) + x)
    for i in range(x.size):
        if Id[i] > 1:
            for k in range(2,int(Id[i])+1):
                f += np.log(k)
    for i in range(x.size):
        f -= Hcubic(x[i])
    return f

# constraint derivative wrt x
def cons_J(x):
    J = -Id/x + 1
    for i in range(x.size):
        J[i] -= dHcubic(x[i])
    return J

# constraint Hessian at x applied to v
def cons_H(x,v):
    Hdiag = Id/x**2
    NN = len(x)
    for i in range(NN):
        Hdiag[i] -= ddHcubic(x[i])
    Harray = dia_array((Hdiag,0),shape=(NN,NN)).toarray() #sparse diagonal matrix
    return Harray*v


# set up constraints
# bounds = Bounds([0.0]*(N*N2), [np.inf]*(N*N2),[True]*(N*N2))
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 0, jac=cons_J, hess=cons_H)

# starting point is current intensity
x0 = I

# minimize
res = minimize(obj, x0, method='trust-constr', jac=obj_der, hess=obj_hess,
               constraints=[nonlinear_constraint],
               options={'verbose': 2, 'maxiter': 1000, 'gtol': 1e-4})
# print PN(I)
print(res.x)
'''
# round solution down to nearest integer for readability
resxrounded = np.array([0]*(N*N2))
for i in range(len(res.x)):
    resxrounded[i] = int(np.floor(res.x[i]))
print(resxrounded)
'''

# Finite difference confirms analytic gradients are correct
'''
# Finite difference gradient check
h = 1e-4
xx = Id+1
diffJ = np.zeros(N*N2)
for i in range(len(diffJ)):
    xxnew = np.copy(xx)
    xxnew[i] += h
    diffJ[i] = (cons_f(xxnew) - cons_f(xx))/h
trueJ = cons_J(xx)
print(diffJ)
print(trueJ)
'''

# Chebyshev interpolation (not as good as cubic spline)
'''
Ncheb = 1000
Hcoeffs = chebyshev.chebinterpolate(Hvec,Ncheb)
dHcoeffs = chebyshev.chebinterpolate(dHvec,Ncheb)
ddHcoeffs = chebyshev.chebinterpolate(ddHvec,Ncheb)

# add shift from [10^-10, 10^4] to [-1,1] back into Hvec, dHvec, ddHvec before running
def Hcheb(x):
    xshift = 2*x/(b-a)-1
    return chebyshev.chebval(xshift,Hcoeffs)

def dHcheb(x):
    xshift = 2*x/(b-a)-1
    return chebyshev.chebval(xshift,dHcoeffs)

def ddHcheb(x):
    xshift = 2*x/(b-a)-1
    return chebyshev.chebval(xshift,ddHcoeffs)

xx = np.linspace(1e-2,1e1,1000)
ddHchebx = ddHcheb(xx)

ddHx = np.zeros_like(xx)
for i in range(len(ddHx)):
    ddHx[i] = ddH(xx[i])

plt.plot(xx,ddHx)
plt.plot(xx,ddHchebx)
plt.show()

xx = np.linspace(1e-10,1e3,1000)
Hx = np.zeros_like(xx)
for i in range(len(Hx)):
    Hx[i] = H(xx[i])

plt.plot(xx,Hx)
plt.plot(xx,Hcubic(xx))
plt.show()
'''
