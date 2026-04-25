import argparse
import numpy as np
import scipy.sparse as sp
import scipy
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
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--func", type=str, default="log")
args = parser.parse_args()

n = args.n
factor = np.pi**2 / 16  # sigma_R^2/4
c = 1000

basis_1d = np.linspace(0, 1, 3 * n + 1)
X, Y = np.meshgrid(basis_1d, basis_1d, indexing="ij")
x_basis, y_basis = X.ravel(), Y.ravel()

if args.func == "log":
    u_func = lambda x, y: x * np.log(y + 1)
    f = -6 * x_basis / (y_basis + 1) ** 4
    grad_func = lambda x, y: (np.log(y + 1), x / (y + 1))
elif args.func == "poly":
    u_func = lambda x, y: x**6 + y**5 + 3 * x**2 * y**3
    f = 360 * x_basis**2 + 192 * y_basis
    grad_func = lambda x, y: (6 * x**5 + 6 * x * y**3, 5 * y**4 + 9 * x**2 * y**2)
elif args.func == "esin":
    u_func = lambda x, y: np.exp(x) * np.sin(y)
    f = 0
    grad_func = lambda x, y: (np.exp(x) * np.sin(y), np.exp(x) * np.cos(y))
elif args.func == "sincos":
    u_func = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
    f = 4 * np.pi**4 * u_func(x_basis, y_basis)
    grad_func = lambda x, y: (
        np.pi * np.cos(np.pi * x) * np.cos(np.pi * y),
        np.pi * -np.sin(np.pi * x) * np.sin(np.pi * y),
    )

for delta in np.logspace(-5, -2, num=50, base=10):
    print("delta:", delta)
    xi = delta / c

    quadrature_nodes, quadrature_weights = build_quadrature(n, delta)
    global_quadrature_nodes = ((np.arange(n) / n)[:, None] + quadrature_nodes[None, :]).ravel()
    global_quadrature_weights = np.tile(quadrature_weights, n)

    Phi = sp.diags(global_quadrature_weights)
    Phi_2d = sp.diags(np.kron(global_quadrature_weights, global_quadrature_weights))
    Phi_simpson = build_simpson_weights(n)
    Phi_simpson_2d = np.kron(Phi_simpson, Phi_simpson)

    A = build_coeffs_matrix(n, quadrature_nodes, delta)

    quadrature_nodes_interp_matrix = build_interp_matrix(n, global_quadrature_nodes)
    f0_values = kernel_integral_0_to_1(delta, global_quadrature_nodes)
    B = sp.diags(f0_values) * quadrature_nodes_interp_matrix

    # Check if everything is correct (optional)
    # assert np.allclose(A.sum(axis=1).A1, B.sum(axis=1).A1)

    A_2d = sp.kron(A, A).tocsr()
    B_2d = sp.kron(B, B).tocsr()

    L_2d = B_2d - A_2d

    # Explicitly compute the kernel-weighted Neumann boundary term.
    # Note that, under the kernel choice in the paper, R_bar differs from R by a factor of 4
    R_bar_0 = np.exp(-(global_quadrature_nodes**2) / delta**2) / delta
    R_bar_1 = np.exp(-((global_quadrature_nodes - 1) ** 2) / delta**2) / delta
    m = len(global_quadrature_nodes)
    R_x0 = sp.csr_matrix((np.repeat(R_bar_0, m), (np.arange(m**2), np.tile(np.arange(m), m))), shape=(m**2, m))
    R_x1 = sp.csr_matrix((np.repeat(R_bar_1, m), (np.arange(m**2), np.tile(np.arange(m), m))), shape=(m**2, m))
    R_y0 = sp.csr_matrix((np.tile(R_bar_0, m), (np.arange(m**2), np.repeat(np.arange(m), m))), shape=(m**2, m))
    R_y1 = sp.csr_matrix((np.tile(R_bar_1, m), (np.arange(m**2), np.repeat(np.arange(m), m))), shape=(m**2, m))
    b_x0 = np.dot(np.column_stack(grad_func(np.zeros_like(basis_1d), basis_1d)), np.array([-1, 0]))
    b_x1 = np.dot(np.column_stack(grad_func(np.ones_like(basis_1d), basis_1d)), np.array([1, 0]))
    b_y0 = np.dot(np.column_stack(grad_func(basis_1d, np.zeros_like(basis_1d))), np.array([0, -1]))
    b_y1 = np.dot(np.column_stack(grad_func(basis_1d, np.ones_like(basis_1d))), np.array([0, 1]))
    R_bar_b = 1 / 2 * ((R_x0 * A * b_x0 + R_x1 * A * b_x1 + R_y0 * A * b_y0 + R_y1 * A * b_y1))

    A_bd_0, A_bd_1 = build_boundary_coeffs_vector(n, delta)
    B_0, B_1 = kernel_integral_0_to_1(delta, 0), kernel_integral_0_to_1(delta, 1)

    A_x0 = sp.kron(A_bd_0, A).tocsr()
    A_x1 = sp.kron(A_bd_1, A).tocsr()
    A_y0 = sp.kron(A, A_bd_0).tocsr()
    A_y1 = sp.kron(A, A_bd_1).tocsr()

    LHS = 1 / delta**4 * L_2d.T * Phi_2d * L_2d + 1 / xi * (A_x0.T * Phi * A_x0 + A_x1.T * Phi * A_x1 + A_y0.T * Phi * A_y0 + A_y1.T * Phi * A_y1)
    RHS = (
        factor * Phi_simpson_2d * f
        + 1 / delta**2 * L_2d.T * Phi_2d * R_bar_b
        + 1
        / xi
        * (
            A_x0.T * Phi * (B_0 * f0_values * u_func(0, global_quadrature_nodes))
            + A_x1.T * Phi * (B_1 * f0_values * u_func(1, global_quadrature_nodes))
            + A_y0.T * Phi * (B_0 * f0_values * u_func(global_quadrature_nodes, 0))
            + A_y1.T * Phi * (B_1 * f0_values * u_func(global_quadrature_nodes, 1))
        )
    )

    if delta < 4.5e-3:
        # It's a parameter that may be dependent on the device.
        # On my laptop, when delta is larger than this, dense solve is faster than sparse solve.
        u_basis = sp.linalg.spsolve(LHS, RHS)
    else:
        u_basis = scipy.linalg.solve(LHS.todense(), RHS, assume_a="sym")

    # Validation
    num_test = 1000
    basis_1d_test = np.linspace(0, 1, num_test, endpoint=False)
    X, Y = np.meshgrid(basis_1d_test, basis_1d_test, indexing="ij")
    x_test, y_test = X.ravel(), Y.ravel()

    test_nodes_interp_matrix = build_interp_matrix(n, basis_1d_test)
    u = sp.kron(test_nodes_interp_matrix, test_nodes_interp_matrix) * u_basis
    error = np.sqrt(np.mean((u - u_func(x_test, y_test)) ** 2))
    print("Error:", error)

    u_basis_reshape = u_basis.reshape(3 * n + 1, 3 * n + 1)
    bd_error = np.sqrt(
        np.sum(
            (test_nodes_interp_matrix * u_basis_reshape[0, :] - u_func(0, basis_1d_test)) ** 2
            + (test_nodes_interp_matrix * u_basis_reshape[-1, :] - u_func(1, basis_1d_test)) ** 2
            + (test_nodes_interp_matrix * u_basis_reshape[:, 0] - u_func(basis_1d_test, 0)) ** 2
            + (test_nodes_interp_matrix * u_basis_reshape[:, -1] - u_func(basis_1d_test, 1)) ** 2
        )
        / (4 * num_test)
    )
    print("Boundary Error:", bd_error)

    grad_coeffs = cubic_lagrange_basis_coeffs(0, n)[:, -2]
    u_grad_x0 = (test_nodes_interp_matrix * u_basis_reshape[:4].T) @ grad_coeffs
    u_grad_x1 = (test_nodes_interp_matrix * u_basis_reshape[-4:].T) @ -np.flip(grad_coeffs)
    u_grad_y0 = (test_nodes_interp_matrix * u_basis_reshape[:, :4]) @ grad_coeffs
    u_grad_y1 = (test_nodes_interp_matrix * u_basis_reshape[:, -4:]) @ -np.flip(grad_coeffs)
    bd_grad_error = np.sqrt(
        np.sum(
            (u_grad_x0 - grad_func(0, basis_1d_test)[0])[1:] ** 2
            + (u_grad_x1 - grad_func(1, basis_1d_test)[0])[1:] ** 2
            + (u_grad_y0 - grad_func(basis_1d_test, 0)[1])[1:] ** 2
            + (u_grad_y1 - grad_func(basis_1d_test, 1)[1])[1:] ** 2
        )
        / (4 * num_test - 4)
    )
    print("Boundary Normal Error:", bd_grad_error)

    print()
