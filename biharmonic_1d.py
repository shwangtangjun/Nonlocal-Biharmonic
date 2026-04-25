import argparse
import numpy as np
import scipy.sparse as sp
from utils import (
    cubic_lagrange_basis_coeffs,
    build_coeffs_matrix,
    build_interp_matrix,
    build_quadrature,
    build_simpson_weights,
    build_boundary_coeffs_vector,
    kernel_integral_0_to_1,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--func", type=str, default="poly")
args = parser.parse_args()


n = args.n
factor = np.pi / 16  # sigma_R^2/4
c = 1000

x_basis = np.linspace(0, 1, 3 * n + 1)

if args.func == "poly":
    p = 10
    u_func = lambda x: x**p
    if p >= 4:
        f = p * (p - 1) * (p - 2) * (p - 3) * x_basis ** (p - 4)
    else:
        f = np.zeros_like(x_basis)
    grad_0 = p * 0 ** (p - 1)
    grad_1 = p * 1 ** (p - 1)
elif args.func == "exp":
    u_func = lambda x: np.exp(x)
    f = np.exp(x_basis)
    grad_0 = 1
    grad_1 = np.exp(1)
elif args.func == "sin":
    u_func = lambda x: np.sin(np.pi * x)
    f = np.pi**4 * np.sin(np.pi * x_basis)
    grad_0 = np.pi
    grad_1 = -np.pi
elif args.func == "log":
    u_func = lambda x: np.log(x + 1)
    f = -6 / (x_basis + 1) ** 4
    grad_0 = 1
    grad_1 = 1 / 2
else:
    raise NotImplementedError

for delta in np.logspace(-5, -2, num=50, base=10):
    print("delta:", delta)
    xi = delta / c

    quadrature_nodes, quadrature_weights = build_quadrature(n, delta)
    global_quadrature_nodes = ((np.arange(n) / n)[:, None] + quadrature_nodes[None, :]).ravel()
    global_quadrature_weights = np.tile(quadrature_weights, n)

    Phi = sp.diags(global_quadrature_weights)
    Phi_simpson = build_simpson_weights(n)

    A = build_coeffs_matrix(n, quadrature_nodes, delta)

    quadrature_nodes_interp_matrix = build_interp_matrix(n, global_quadrature_nodes)
    f0_values = kernel_integral_0_to_1(delta, global_quadrature_nodes)
    B = sp.diags(f0_values) * quadrature_nodes_interp_matrix

    # Check if everything is correct (optional)
    # assert np.allclose(A.sum(axis=1).A1, B.sum(axis=1).A1)

    L = B - A

    # Explicitly compute the kernel-weighted Neumann boundary term.
    # Note that, under the kernel choice in the paper, R_bar differs from R by a factor of 4
    R_bar_0 = np.exp(-(global_quadrature_nodes**2) / delta**2) / delta
    R_bar_1 = np.exp(-((global_quadrature_nodes - 1) ** 2) / delta**2) / delta
    R_bar_b = 1 / 2 * (R_bar_0 * grad_0 * -1 + R_bar_1 * grad_1 * 1)

    A_bd_0, A_bd_1 = build_boundary_coeffs_vector(n, delta)
    B_0, B_1 = kernel_integral_0_to_1(delta, 0), kernel_integral_0_to_1(delta, 1)

    LHS = 1 / delta**4 * L.T * Phi * L + 1 / xi * (np.outer(A_bd_0, A_bd_0) + np.outer(A_bd_1, A_bd_1))
    RHS = factor * Phi_simpson * f + 1 / delta**2 * L.T * Phi * R_bar_b + 1 / xi * (A_bd_0 * B_0 * u_func(0) + A_bd_1 * B_1 * u_func(1))

    u_basis = sp.linalg.spsolve(LHS, RHS)

    # Validation
    num_test = 5000
    x_test = np.linspace(0, 1, num_test, endpoint=False)
    test_nodes_interp_matrix = build_interp_matrix(n, x_test)
    error = np.sqrt(np.mean((test_nodes_interp_matrix * u_basis - u_func(x_test)) ** 2))
    print("Error:", error)

    bd_error = np.sqrt(1 / 2 * ((u_basis[0] - u_func(0)) ** 2 + (u_basis[-1] - u_func(1)) ** 2))
    print("Boundary Error:", bd_error)

    grad_coeffs = cubic_lagrange_basis_coeffs(0, n)[:, -2]  # 1st order coeffs
    u_grad_0 = np.dot(u_basis[:4], grad_coeffs)
    u_grad_1 = np.dot(u_basis[-4:], -np.flip(grad_coeffs))  # symmetry
    bd_grad_error = np.sqrt(1 / 2 * ((u_grad_0 - grad_0) ** 2 + (u_grad_1 - grad_1) ** 2))
    print("Boundary Normal Error:", bd_grad_error)
    print()
