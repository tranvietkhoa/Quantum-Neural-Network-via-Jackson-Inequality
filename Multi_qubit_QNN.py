import math
import cmath
from random import uniform
from scipy.integrate import quad, nquad, dblquad, tplquad
import numpy as np
from numpy import linalg
from scipy import signal
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

epsl = 1e-10

from qiskit import QuantumCircuit, transpile
#from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator

class MulTriMono:
    def __init__(self, degrees, coefficient):
        self.degrees = degrees
        self.coefficient = coefficient

    def get_degrees(self):
        return self.degrees

    def get_coefficient(self):
        return self.coefficient

    def evaluate(self, x):
        result = self.coefficient
        for i in range(len(self.degrees)):
            result *= np.exp(complex(0,1) * self.degrees[i] * x[i])
        return result

class MulTriPoly:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def get_coefficients(self):
        return self.coefficients

    def get_coefficient(self, degree):
        return self.coefficients[degree]

    def evaluate(self, x):
        result = 0
        for degree in self.coefficients:
            result += MulTriMono(degree, self.coefficients[degree]).evaluate(x)
        return result

    def add_monomial(self, mono):
        if mono.get_degrees() in self.coefficients:
            self.coefficients[mono.get_degrees()] = self.coefficients[mono.get_degrees()] + mono.get_coefficient()
        else:
            self.coefficients[mono.get_degrees()] = mono.get_coefficient()
            
def main_function_Jkernel(N, r, t):
    if (t == 0):
        return pow(math.floor(N/2) + 1, 2 * r)
    return pow(np.sin((t/2) * (1 + math.floor(N/2))) / np.sin(t/2) , 2 * r)

def find_lambda(N, r): # we can pre-compute this
    func = lambda t: main_function_Jkernel(N, r, t)
    return quad(func, -math.pi, math.pi, limit=200)[0]

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

def tuplem_nK(n, N, K):
    result = 1
    for j in range(len(N)):
        result *= m_nK(n[j], N[j], K[j])
    return result

def fourier_coef(f, at, Nx = 256, Ny = 256): # implement FFT
    x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    func = f(X, Y)
    F = np.fft.fft2(func)
    F = F / (Nx * Ny)

    kx = np.fft.fftfreq(Nx, 1/Nx).astype(int)
    ky = np.fft.fftfreq(Ny, 1/Ny).astype(int)

    i = np.where(kx == at[0])[0][0]
    j = np.where(ky == at[1])[0][0]
    return F[i, j]

def fourier_heat(g, at, t):
    norm = sum([at_i*at_i for at_i in at])
    return np.exp(-norm * t) * fourier_coef(g, at)

from itertools import product

def lattice_points(N):
    ranges = [range(-Nj, Nj+1) for Nj in N]
    return list(product(*ranges))

def mathcalT(f, N, K):
    ld = 1
    tup = ()
    for j in range(len(N)):
        ld *= find_lambda(N[j], math.ceil((K[j]+3)/2))
        tup = tup + (math.ceil((K[j]+3)/2) * math.floor(N[j]/2),)
    trig_poly = MulTriPoly({})

    for n in lattice_points(tup):
        trig_poly.add_monomial(MulTriMono(n, fourier_coef(f, n) * tuplem_nK(n, N, K) * pow(2*math.pi, len(N)) / ld))
    return trig_poly

def mathcalT_heat(g, t, N, K):
    ld = 1
    tup = ()
    for j in range(len(N)):
        ld *= find_lambda(N[j], math.ceil((K[j]+3)/2))
        tup = tup + (math.ceil((K[j]+3)/2) * math.floor(N[j]/2),)
    trig_poly = MulTriPoly({})

    for n in lattice_points(tup):
        trig_poly.add_monomial(MulTriMono(n, fourier_heat(g, n, t) * tuplem_nK(n, N, K) * pow(2*math.pi, len(N)) / ld))
    return trig_poly

def create_unitary_matrix(v):
    v = np.array(v, dtype=float).flatten()
    norm_v = np.linalg.norm(v)
    n = len(v)
    v1 = v / norm_v
    U = np.zeros((n, n))
    U[:, 0] = v1
    basis_vectors = np.eye(n)
    if np.allclose(v1, basis_vectors[:, 0]):
        basis_vectors = np.roll(basis_vectors, 1, axis=1)

    for i in range(1, n):
        u = basis_vectors[:, i]
        for j in range(i):
            u = u - np.dot(U[:, j], u) * U[:, j]
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-10:
            raise ValueError("Cannot construct unitary matrix: vectors are linearly dependent")
        U[:, i] = u / norm_u

    return U

def make_VN2(q):
    size = pow(2, q)
    v = np.array([1/math.sqrt(size) for i in range(size)])
    return create_unitary_matrix(v)

def precompute_VN():
    result = []
    for q in range(1, 12):
        result.append(make_VN2(q))
    return result

precomputed_VN = precompute_VN()

def make_VN(q):
    return precomputed_VN[q-1]

def make_CN2(all_U, d, q):
    C_N = np.zeros((pow(2, d+q), pow(2, d+q)), dtype=complex)
    size = pow(2, q)
    for i in range(size):
        vi = np.array([1 if j == i else 0 for j in range(size)])
        C_N += np.kron(np.outer(vi, vi), all_U[i])
    return C_N

def make_CN(all_U, d, q):
    C_N = np.zeros((pow(2, d+q), pow(2, d+q)), dtype=complex)
    size = pow(2, q)
    for i in range(size):
        for j in range(pow(2,d)):
            for k in range(pow(2,d)):
                C_N[pow(2,d)*i + j][pow(2,d)*i + k] = all_U[i][j][k]
    return C_N

def make_U0(d):
    deg = (0,0)
    if d == 3:
        deg = (0,0,0)
    para = find_parameters_from_monomial(MulTriMono(deg, 0))
    qc = QuantumCircuit(d)

    for i in range(d):
        qc.rz(para[i][1][-1], i)
        qc.ry(para[i][0][0], i)
        qc.rz(para[i][1][0], i)

        for j in range(1, len(para[i][0])):
            qc.rz(x, i)
            qc.ry(para[i][0][j], i)
            qc.rz(para[i][1][j], i)
    U0 = Operator(qc).data
    return U0

def make_CN2(all_U, d, q):
    C_N = np.zeros((pow(2, d+q), pow(2, d+q)), dtype=complex)
    size = pow(2, q)
    for i in range(size):
        for j in range(pow(2,d)):
            for k in range(pow(2,d)):
                C_N[pow(2,d)*i + j][pow(2,d)*i + k] = all_U[i][j][k]
    return C_N

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
    A = (P * P.conjugate()).minus_by_one() #oneminus(multiply(P, P.conjugate()))
    deg = A.get_degree()

    while (deg >= 0):
        if abs(A.get_coefficient()[deg]) < epsl:
            deg -= 1
        else:
            break
    if deg == -1:
        return LaurentPoly(0, [0])
    coefs = [0] * (2*deg + 1)
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
    if np.round((Q * Q.conjugate() + P * P.conjugate()).get_coefficient()[0]) - 1 > epsl:
        print("Warning: Incompatible value of N. Try restricting N <= 27")
    return Q

def find_parameters(P, Q):
    M = P.get_degree()
    theta = [0] * (M+1)
    phi = [0] * (M+2)

    if Q.get_degree() == 0 and abs(Q.get_coefficient()[0]) < epsl and abs(P.get_coefficient()[M]) > 0: # and power is pos
        if M != 0:
            theta[0] = 3 * math.pi
            theta[M] = math.pi
        return theta, phi

    if Q.get_degree() == 0 and abs(Q.get_coefficient()[0]) < epsl and abs(P.get_coefficient()[M]) < epsl: # and power is neg
        return theta, phi


    if Q.get_degree() == 0 and abs(Q.get_coefficient()[0] > epsl) and abs(P.get_coefficient()[M]) < epsl: # power is neg and const is non zero
        theta[0] = 2 * math.acos(abs(P.get_coefficient()[-M]))
        phi[0] = cmath.phase(Q.get_coefficient()[0]) - cmath.phase(P.get_coefficient()[-M])
        phi[-1] = - cmath.phase(Q.get_coefficient()[0]) - cmath.phase(P.get_coefficient()[-M])
        return theta, phi

    newP, newQ = P, Q

    # power is pos and const is non zero
    for i in range(M, 0, -1):

        if i > M/2:
            if i == M:
                theta[i] = math.pi
                newP, newQ = LaurentPoly(1, [0, Q.get_coefficient()[0], 0]), (newP * LaurentPoly(1, [0, 0, -1])).lower_deg_by_2()
            else:
                newP, newQ = newP * LaurentPoly(1, [0, 1, 0]), (newQ * LaurentPoly(1, [0, 0, 1])).lower_deg_by_2()
        elif i == M/2:

            z = (-newP.get_coefficient()[i]) / newQ.get_coefficient()[i]
            theta[i] = 2*math.atan(abs(z))
            phi[i] = -cmath.phase(z / math.tan(theta[i]/2))


            newP, newQ = (newP * LaurentPoly(1, [0, np.exp(complex(0,1)*phi[i]/2) * np.cos(theta[i]/2), 0]) + newQ * LaurentPoly(1, [0, np.exp( - complex(0,1)*phi[i]/2) * np.sin(theta[i]/2), 0])).lower_deg_by_2(), (newP * LaurentPoly(1, [0, 0, -np.exp(complex(0,1)*phi[i]/2) * np.sin(theta[i]/2)]) + newQ * LaurentPoly(1, [0, 0, np.exp(-complex(0,1)*phi[i]/2) * np.cos(theta[i]/2)])).lower_deg_by_2()

            tem = - newQ.get_coefficient()[i-1]

            theta[0] = 3*math.pi
            phi[0] = cmath.phase(tem)
            phi[-1] = - cmath.phase(tem)

            return theta, phi

    theta[0] = 2 * math.acos(abs(P.get_coefficient()[-M]))
    phi[0] = cmath.phase(Q.get_coefficient()[0]) - cmath.phase(P.get_coefficient()[-M])
    phi[-1] = - cmath.phase(Q.get_coefficient()[0]) - cmath.phase(P.get_coefficient()[-M])
    return theta, phi

def find_parameters_from_monomial(t):
    para = []
    for i in range(len(t.get_degrees())):
        deg = t.get_degrees()[i] * 2
        coefs = [0] * (abs(deg) * 2 + 1)
        if i == 0:
            coefs[deg] = t.get_coefficient()
        else:
            coefs[deg] = 1
        P = LaurentPoly(abs(deg), coefs)
        Q = find_Q(P)
        para.append(find_parameters(P, Q))
    return para

def find_parameters_from_polynomial(T):
    para = {}
    for degree in T.get_coefficients():
        t = MulTriMono(degree, T.get_coefficient(degree))
        para[degree] = find_parameters_from_monomial(t)
    return para

def multiqbqnn(trig_poly, x):
    d = len(x)

    all_U = []
    all_paras = find_parameters_from_polynomial(trig_poly)

    tt = time.time()
    for degree in all_paras:

        qc = QuantumCircuit(d)
        para = all_paras[degree]
        for i in range(d):

            qc.rz(para[i][1][-1], i)
            qc.ry(para[i][0][0], i)
            qc.rz(para[i][1][0], i)

            for j in range(1, len(para[i][0])):
                qc.rz(x[i], i)
                qc.ry(para[i][0][j], i)
                qc.rz(para[i][1][j], i)
        U_n = Operator(qc).data
        all_U.append(U_n)


    N = len(all_U)
    q = math.ceil(math.log2(N))

    for i in range(N, pow(2, q)):
        all_U.append(make_U0(d))

    CN = make_CN(all_U, d, q)
    VN = make_VN(q)

    qc = QuantumCircuit(d+q)
    ancilla = [i for i in range(d, d+q)]

    qc.append(Operator(VN), ancilla)
    qc.append(Operator(CN), [i for i in range(0, q+d)])
    qc.append(Operator(VN.conj().T), ancilla)

    sv = Statevector.from_instruction(qc)

    return sv.data[0] * pow(2, q)

def g(x1, x2):
    x1_mod = x1 % (2.0*np.pi)
    x2_mod = x2 % (2.0*np.pi)
    g1 = ((x1_mod > 0.0) & (x1_mod < np.pi)).astype(np.float32) - ((x1_mod > np.pi) & (x1_mod < 2.0*np.pi)).astype(np.float32)
    g2 = ((x2_mod > 0.0) & (x2_mod < np.pi)).astype(np.float32) - ((x2_mod > np.pi) & (x2_mod < 2.0*np.pi)).astype(np.float32)
    return g1*g2

def heat_solution_fft(g_func, t, x_point, N=128):

    x1, x2 = x_point

    grid = np.linspace(0, 2*np.pi, N, endpoint=False)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    G = g_func(X, Y)

    G_hat = np.fft.fft2(G) / (N**2)
    k = np.fft.fftfreq(N, d=1.0/N)  
    KX, KY = np.meshgrid(k, k, indexing="ij")

    decay = np.exp(-(KX**2 + KY**2) * t)

    phase = np.exp(1j * (KX*x1 + KY*x2))
    u_val = np.sum(G_hat * decay * phase)

    return np.real(u_val)

N = 7
K = 2
M = 51
x = np.linspace(0, 2.0*np.pi, M)
y = np.linspace(0, 2.0*np.pi, M)
X, Y = np.meshgrid(x, y, indexing="ij")

t = 0.5
u_true = np.zeros([M, M])
u_QNN = np.zeros([K+1, N-1, M, M])
err = np.zeros([K+1, N-1])
for i in range(M):
    for j in range(M):
        u_true[i, j] = heat_solution_fft(g, t, (X[i, j], Y[i, j]))
        
for k in range(K+1):
    for n in range(N-1):
        b = time.time()
        trig_poly = mathcalT_heat(g, t, (n+2, n+2), (k, k))
        for i in range(M):
            for j in range(M):
                u_QNN[k, n, i, j] = multiqbqnn(trig_poly, (X[i, j], Y[i, j])).real
                
        e = time.time()
        print("Heat QNN, time " + str(np.round(e-b, 1)) + "s, K = " + str(k) + "/" + str(K) + ", N = " + str(n+2) + "/" + str(N))
        
for k in range(K+1):
    for n in range(N-1):
        err[k, n] = np.max(np.abs(u_true - u_QNN[k, n]))
        
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
cols = [cmap(i/(K+1)) for i in range(K+1)]
fig = plt.figure()
for k in range(K+1):
    plt.plot(np.arange(2, N+1), err[k], label = "K = " + str(k), color = cols[k])
    
plt.xlabel("$N$ := $N_1$ = $N_2$")
plt.xticks(ticks = np.arange(2, N+1), labels = [str(i) if i % 2 == 0 else '' for i in range(2, N+1)])
plt.ylabel("$\\Vert f - f^{\\mathbf{L}_{\\mathbf{N},\\mathbf{K}}}_{\\mathbf{\\theta},\\mathbf{\\phi}} \\Vert_{\\infty}$")
plt.title("Error rate")
plt.legend()
plt.savefig("QNN_multi_05_err.png", dpi = 500) 
plt.close()

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_wireframe(X, Y, u_true, rstride = 6, cstride = 6, color = "black", linewidth = 0.5)
ax.plot_surface(X, Y, u_QNN[-1, -1], cmap = plt.cm.coolwarm, alpha = 0.7, linewidth = 0, antialiased = False)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$u(0.5,\\mathbf{x})$ and $f^{\\mathbf{L}_{\\mathbf{N},\\mathbf{K}}}_{\\mathbf{\\theta},\\mathbf{\\phi}}(\\mathbf{x})$")
ax.set_zlim([-0.6, 0.6])
plt.title("Function approximation $t=0.5$")
plt.savefig("QNN_multi_05_app.png", dpi = 500) 
plt.close()

t = 1.0
u_true = np.zeros([M, M])
u_QNN = np.zeros([K+1, N-1, M, M])
err = np.zeros([K+1, N-1])
for i in range(M):
    for j in range(M):
        u_true[i, j] = heat_solution_fft(g, t, (X[i, j], Y[i, j]))
        
for k in range(K+1):
    for n in range(N-1):
        b = time.time()
        trig_poly = mathcalT_heat(g, t, (n+2, n+2), (k, k))
        for i in range(M):
            for j in range(M):
                u_QNN[k, n, i, j] = multiqbqnn(trig_poly, (X[i, j], Y[i, j])).real
                
        e = time.time()                
        print("Heat QNN, time " + str(np.round(e-b, 1)) + "s, K = " + str(k) + "/" + str(K) + ", N = " + str(n+2) + "/" + str(N))
        
for k in range(K+1):
    for n in range(N-1):
        err[k, n] = np.max(np.abs(u_true - u_QNN[k, n]))
        
cols = [cmap(i/(K+1)) for i in range(K+1)]
fig = plt.figure()
for k in range(K+1):
    plt.plot(np.arange(2, N+1), err[k], label = "K = " + str(k), color = cols[k])
    
plt.xlabel("$N$ := $N_1$ = $N_2$")
plt.xticks(ticks = np.arange(2, N+1), labels = [str(i) if i % 2 == 0 else '' for i in range(2, N+1)])
plt.ylabel("$\\Vert f - f^{\\mathbf{L}_{\\mathbf{N},\\mathbf{K}}}_{\\mathbf{\\theta},\\mathbf{\\phi}} \\Vert_{\\infty}$")
plt.title("Error rate")
plt.legend()
plt.savefig("QNN_multi_1_err.png", dpi = 500) 
plt.close()

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_wireframe(X, Y, u_true, rstride = 6, cstride = 6, color = "black", linewidth = 0.5)
ax.plot_surface(X, Y, u_QNN[-1, -1], cmap = plt.cm.coolwarm, alpha = 0.7, linewidth = 0, antialiased = False)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$u(1,\\mathbf{x})$ and $f^{\\mathbf{L}_{\\mathbf{N},\\mathbf{K}}}_{\\mathbf{\\theta},\\mathbf{\\phi}}(\\mathbf{x})$")
ax.set_zlim([-0.6, 0.6])
plt.title("Function approximation $t=1$")
plt.savefig("QNN_multi_1_app.png", dpi = 500) 
plt.close()