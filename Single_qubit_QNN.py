import math
import cmath
from random import uniform
from scipy.integrate import quad
import numpy as np
from numpy import linalg
from scipy import signal
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from qiskit import QuantumCircuit, transpile
#from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = list(coefficients)
        self.degree = len(coefficients) - 1

    def get_degree(self):
        return self.degree

    def get_coefficient(self):
        return self.coefficients

    def evaluate(self, x):
        result = 0
        for i in range(self.degree + 1):
            result += self.coefficients[i] * math.pow(x, i)
        return result

    def find_roots(self):
        return np.roots(self.coefficients[::-1])

    def __repr__(self):
        terms = []
        for i, coef in enumerate(self.coefficients):
            if coef != 0:
                if i == 0:
                    terms.append(f"{coef}")
                elif i == 1:
                    terms.append(f"{coef}x")
                else:
                    terms.append(f"{coef}x^{i}")
        return " + ".join(terms) if terms else "0"

    def __mul__(self, other):
        deg_P = self.degree
        deg_Q = other.degree
        coefs_P = self.coefficients
        coefs_Q = other.coefficients

        result_coefs = [0] * (deg_P + deg_Q + 1)
        for i in range(deg_P + 1):
            for j in range(deg_Q + 1):
                result_coefs[i + j] += coefs_P[i] * coefs_Q[j]

        return Polynomial(result_coefs)
    
class TriPoly:
    def __init__(self, degree, coefficients):
        self.coefficients = coefficients
        self.degree = degree

    def get_degree(self):
        return self.degree

    def get_coefficient(self):
        return self.coefficients

    def evaluate(self, x):
        result = self.coefficients[0]
        for i in range(1, self.degree + 1):
            result += self.coefficients[i] * np.exp(complex(0, 1) * i * x)
            result += self.coefficients[-i] * np.exp(-(complex(0, 1) * i * x))
        return result
    
def main_function_Jkernel(N, r, t):
    if (t == 0):
        return pow(math.floor(N/2) + 1, 2 * r)
    return pow(np.sin((t/2) * (1 + math.floor(N/2))) / np.sin(t/2) , 2 * r)

def find_lambda(N, r): # we can pre-compute this
    func = lambda t: main_function_Jkernel(N, r, t)
    return quad(func, -math.pi, math.pi, limit=200)[0]

def Jkernel(N, r, t):
    return main_function_Jkernel(N, r, t) / find_lambda(N, r)

def mathcalT(f, N, K, x):
    r = math.ceil((K + 3)/2)
    func = lambda t: Jkernel(N, r, t) * sum(math.pow(-1, k+1) * math.comb(K+1, k) * f(x + k*t) for k in range(1, K+2))
    return quad(func, -math.pi, math.pi, limit=200)[0]

def Fourier_coef(f, n):
    func_real = lambda x: np.real(f(x) * np.exp(-complex(0,1) * n * x))
    func_imag = lambda x: np.imag(f(x) * np.exp(-complex(0,1) * n * x))
    return (quad(func_real, -math.pi, math.pi)[0] + complex(0,1) * quad(func_imag, -math.pi, math.pi)[0])  / (2 * math.pi)

def count_vectors(r, a, b, n): # count vectors z in \mathbb{Z}^r such that a <= z_i <= b and norm_1(z) = n
    if int(2*a) % 2:
        a = int(a + 0.5)
        b = int(b + 0.5)
        n = n + int(r / 2)
    else:
        a = int(a)
        b = int(b)
    if r <= 0:
        return 1 if n == 0 else 0
    m = b - a
    S = n - r * a
    if S < 0 or S > r * m:
        return 0
    total = 0
    max_j = S // (m + 1)
    for j in range(0, max_j + 1):
        sign = -1 if (j % 2 == 1) else 1
        top = S - j * (m + 1) + r - 1
        if top < r - 1:
            continue
        term = math.comb(r, j) * math.comb(top, r - 1)
        total += sign * term
    return total

def m_l(r, l, N):
    return count_vectors(2*r, -math.floor(N/2)/2, math.floor(N/2)/2, l)

def m_nK(n, N, K):
    result = 0
    r = math.ceil((K + 3)/2)
    k = 1
    while (k <= K+1 and k*abs(n) <= r*math.floor(N/2)):
        result += pow(-1, k+1) * math.comb(K+1, k) * m_l(r, k*n, N)
        k += 1
    return result

def trig_mathcalT(f, N, K):
    r = math.ceil((K + 3)/2)
    coefs = [0] * (2*r*math.floor(N/2) + 1)
    n = -r*math.floor(N/2)
    while(n <= r*math.floor(N/2)):
        coefs[n] = m_nK(n, N, K) * Fourier_coef(f, n) * 2 * math.pi / find_lambda(N, r)
        n += 1

    return TriPoly(r*math.floor(N/2), coefs)

class LaurentPoly:
    def __init__(self, degree, coefficients):
        self.coefficients = coefficients
        self.degree = degree

    def get_degree(self):
        return self.degree

    def get_coefficient(self):
        return self.coefficients

    def evaluate(self, x):
        result = self.coefficients[0]
        for i in range(1, self.degree + 1):
            result += self.coefficients[i] * np.exp(complex(0, 1) * i * x / 2)
            result += self.coefficients[-i] * np.exp(-(complex(0, 1) * i * x) / 2)
        return result

    def conjugate(self):
        coefs = [element.conjugate() for element in self.coefficients]
        for n in range(1, self.get_degree() + 1):
            tem = coefs[n]
            coefs[n] = coefs[-n]
            coefs[-n] = tem
        return LaurentPoly(self.degree, coefs)

    def rescale(self, c):
        coefs = [element / c for element in self.coefficients]
        return LaurentPoly(self.degree, coefs)

    def lower_deg_by_2(self):
        deg = self.degree
        new_deg = deg - 2
        new_coefs = [0] * (new_deg * 2 + 1)
        for i in range(-new_deg, new_deg + 1):
            new_coefs[i] = self.coefficients[i]
        return LaurentPoly(new_deg, new_coefs)

    def minus_by_one(self):
        coefs = [0] * (self.degree * 2 + 1)
        for i in range(- self.degree, self.degree + 1):
            coefs[i] = - self.coefficients[i]
            if i == 0:
                coefs[i] += 1
        return LaurentPoly(self.degree, coefs)

    def __add__(self, other):
        P = self
        Q = other
        if Q.get_degree() < P.get_degree():
            P, Q = Q, P
        coefs = Q.get_coefficient()
        for i in range(- P.get_degree(), P.get_degree() + 1):
            coefs[i] += P.get_coefficient()[i]
        return LaurentPoly(Q.get_degree(), coefs)

    def __mul__(self, other):
        P = self
        Q = other
        coefs = [0] * ((P.get_degree() + Q.get_degree())*2 + 1)
        for i in range(- P.get_degree(), P.get_degree() + 1):
            for j in range(- Q.get_degree(), Q.get_degree() + 1):
                coefs[i+j] += P.get_coefficient()[i] * Q.get_coefficient()[j]
        return LaurentPoly(P.get_degree() + Q.get_degree(), coefs)
    
def convert_TriPoly_to_LaurentPoly(P):
    deg = 2*P.get_degree()
    coefs = [0] * (2*deg + 1)
    for n in range(-P.get_degree(), P.get_degree()+1):
        coefs[2*n] = P.get_coefficient()[n]
    return LaurentPoly(deg, coefs)

def add(P, Q):
    if Q.get_degree() < P.get_degree():
        P, Q = Q, P
    coefs = Q.get_coefficient()
    for i in range(- P.get_degree(), P.get_degree() + 1):
        coefs[i] += P.get_coefficient()[i]
    return LaurentPoly(Q.get_degree(), coefs)

def multiply(P, Q):
    coefs = [0] * ((P.get_degree() + Q.get_degree())*2 + 1)
    for i in range(- P.get_degree(), P.get_degree() + 1):
        for j in range(- Q.get_degree(), Q.get_degree() + 1):
            coefs[i+j] += P.get_coefficient()[i] * Q.get_coefficient()[j]
    return LaurentPoly(P.get_degree() + Q.get_degree(), coefs)

def oneminus(P):
    coefs = [0] * (P.get_degree() * 2 + 1)
    for i in range(- P.get_degree(), P.get_degree() + 1):
        coefs[i] = - P.get_coefficient()[i]
        if i == 0:
            coefs[i] += 1
    return LaurentPoly(P.get_degree(), coefs)

def lower_deg_by_2(P):
    deg = P.get_degree()
    new_deg = deg - 2
    new_coefs = [0] * (new_deg * 2 + 1)
    for i in range(-new_deg, new_deg + 1):
        new_coefs[i] = P.get_coefficient()[i]
    return LaurentPoly(new_deg, new_coefs)

def pair_conjugates(roots):
    result = []
    matched = [0] * len(roots)
    for i in range(len(roots)):
        if matched[i]:
            continue
        matched[i] = 1
        simi_score = 100
        for j in range(len(roots)):
            if matched[j]:
                continue
            if abs(1/roots[i].conjugate() - roots[j]) < simi_score:
                cur_ind = j
                simi_score = abs(1/roots[i].conjugate() - roots[j])
        result.append([roots[i], roots[cur_ind]])
        matched[cur_ind] = 1
    return result

def find_Q(P):
    A = oneminus(multiply(P, P.conjugate()))
    deg = A.get_degree()
    coefs = [0] * (2*deg + 1)
    #print(A.get_coefficient()[deg])
    for n in range(2*deg + 1):
        coefs[n] = A.get_coefficient()[-deg + n] / A.get_coefficient()[deg]
    B = Polynomial(coefs)
    roots = B.find_roots()
    conj_pairs = pair_conjugates(roots)
    const = A.get_coefficient()[deg]
    C = Polynomial([1])
    for i in range(deg):
        const *= conj_pairs[i][0]
        C = C * Polynomial([-conj_pairs[i][1], 1])
    const = math.sqrt(const.real)
    coefs_Q = [0] * (deg + 1)
    for n in range(-int(deg/2), int(deg/2) + 1):
        coefs_Q[n] = C.get_coefficient()[n+int(deg/2)] * const
    Q = LaurentPoly(int(deg/2), coefs_Q)
    #if np.round(add(multiply(Q, Q.conjugate()), multiply(P, P.conjugate())).get_coefficient()[0]) != 1:
        #print("Warning: Incompatible value of N. Try restricting N <= 27")
    return Q

def find_parameters(P, Q):
    M = P.get_degree()
    theta = [0] * (M+1)
    phi = [0] * (M+2)
    newP, newQ = P, Q
    for i in range(M, 0, -1):
        z = (-newP.get_coefficient()[i]) / newQ.get_coefficient()[i]
        theta[i] = 2*math.atan(abs(z))
        phi[i] = -cmath.phase(z / math.tan(theta[i]/2))
        newP, newQ = lower_deg_by_2(add(multiply(newP, LaurentPoly(1, [0, np.exp(complex(0,1)*phi[i]/2) * np.cos(theta[i]/2), 0])), multiply(newQ , LaurentPoly(1, [0, np.exp( - complex(0,1)*phi[i]/2) * np.sin(theta[i]/2), 0])))), lower_deg_by_2(add(multiply(newP, LaurentPoly(1, [0, 0, -np.exp(complex(0,1)*phi[i]/2) * np.sin(theta[i]/2)])), multiply(newQ , LaurentPoly(1, [0, 0, np.exp(-complex(0,1)*phi[i]/2) * np.cos(theta[i]/2)]))))

    theta[0] = 2 * math.acos(abs(newP.get_coefficient()[0]))
    phi[0] = cmath.phase(newQ.get_coefficient()[0]) - cmath.phase(newP.get_coefficient()[0])
    phi[-1] = - cmath.phase(newQ.get_coefficient()[0]) - cmath.phase(newP.get_coefficient()[0])
    return theta, phi

def build_circuit(para, x):
    qc = QuantumCircuit(1)

    qc.rz(para[1][-1], 0)
    qc.ry(para[0][0], 0)
    qc.rz(para[1][0], 0)

    for i in range(1, len(para[0])):
        qc.rz(x, 0)
        qc.ry(para[0][i], 0)
        qc.rz(para[1][i], 0)

    return qc

def qc_output(f, x, N, K):
    trigTf = trig_mathcalT(f, N, K)
    P = convert_TriPoly_to_LaurentPoly(trigTf)
    Q = find_Q(P)
    para = find_parameters(P,Q)
    qc = build_circuit(para, x)
    return Statevector.from_instruction(qc).data[0]

N = 20
K = 5
M = 301
x = np.linspace(0.0, 2.0*np.pi, M)
y_QNN = np.zeros([K+1, N, M])
f = lambda x: np.abs(np.sin(x))
err = np.zeros([K+1, N])
for k in range(K+1):
    for n in range(N):
        b = time.time()
        for m in range(M):
            y_QNN[k, n, m] = qc_output(f, x[m], n+1, k).real
            
        e = time.time()            
        err[k, n] = np.max(np.abs(f(x) - y_QNN[k, n]))
        print("f(x)=|sin(x)|, time " + str(np.round(e-b, 1)) + "s, K = " + str(k) + "/" + str(K) + ", N = " + str(n+1) + "/" + str(N))
        
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = [cmap(i/(K+1)) for i in range(K+1)]
fig = plt.figure()
for k in range(K+1):
    plt.plot(np.arange(1, N+1), err[k], label = "K = " + str(k), color = cols[k])
    
plt.xlabel("N")
plt.xticks(ticks = np.arange(1, N+1), labels = [str(i) if i % 2 == 0 else '' for i in range(1, N+1)])
plt.ylabel("$\\Vert f - f^{2L}_{\\theta,\\phi} \\Vert_{\\infty}$")
plt.title("Error rate")
plt.legend(ncol = 3)
plt.savefig("QNN_uni_1_err.png", dpi = 500) 
plt.close()

fig = plt.figure()
plt.plot(x, f(x), color = "black")
for k in range(K+1):
    plt.plot(x, y_QNN[k, -1], label = "QNN K = " + str(k), color = cols[k], linestyle = "dotted")
    
plt.plot(np.nan, np.nan, label = "True", color = "black")
plt.xlabel("x")
plt.ylabel("$f(x) = |\\sin(x)|$ and $f^{2L}_{\\theta,\\phi}(x)$")
plt.ylim(-0.05, 1.25)
plt.title("Function approximation")
plt.legend(ncol = 4, loc = 'upper right', fontsize = 'small')
plt.savefig("QNN_uni_1_app.png", dpi = 500) 
plt.close()

y_QNN = np.zeros([K+1, N, M])
f = lambda x: np.power(np.abs(np.sin(x)), 2.5)
err = np.zeros([K+1, N])
for k in range(K+1):
    for n in range(N):
        b = time.time()
        for m in range(M):
            y_QNN[k, n, m] = qc_output(f, x[m], n+1, k).real
            
        e = time.time()           
        err[k, n] = np.max(np.abs(f(x) - y_QNN[k, n]))
        print("f(x)=|sin(x)|^{2.5}, time " + str(np.round(e-b, 1)) + "s, K = " + str(k) + "/" + str(K) + ", N = " + str(n+1) + "/" + str(N))

fig = plt.figure()
for k in range(K+1):
    plt.plot(np.arange(1, N+1), err[k], label = "K = " + str(k), color = cols[k])
    
plt.xlabel("N")
plt.xticks(ticks = np.arange(1, N+1), labels = [str(i) if i % 2 == 0 else '' for i in range(1, N+1)])
plt.ylabel("$\\Vert f - f^{2L}_{\\theta,\\phi} \\Vert_{\\infty}$")
plt.title("Error rate")
plt.legend(ncol = 3)
plt.savefig("QNN_uni_2_err.png", dpi = 500) 
plt.close()

fig = plt.figure()
plt.plot(x, f(x), color = "black")
for k in range(K+1):
    plt.plot(x, y_QNN[k, -1], label = "QNN K = " + str(k), color = cols[k], linestyle = "dotted")
    
plt.plot(np.nan, np.nan, label = "True", color = "black")
plt.xlabel("x")
plt.ylabel("$f(x) = |\\sin(x)|^{2.5}$ and $f^{2L}_{\\theta,\\phi}(x)$")
plt.ylim(-0.05, 1.25)
plt.title("Function approximation")
plt.legend(ncol = 4, loc = 'upper right', fontsize = 'small')
plt.savefig("QNN_uni_2_app.png", dpi = 500) 
plt.close()