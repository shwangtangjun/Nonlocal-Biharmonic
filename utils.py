import numpy as np
from scipy.special import erf
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss


def cubic_lagrange_basis_coeffs(left, n):
    """
    Return the coefficients of cubic Lagrange basis polynomials on one cell [left, left + 1 / n],
    with four interpolation nodes

        left,
        left + 1 / (3 * n),
        left + 2 / (3 * n),
        left + 1 / n.

    Parameters
    ----------
    left : float
        Left endpoint of the cell.
    n : int
        Number of cells with cell size 1 / n.

    Returns
    -------
    coeff : np.ndarray of shape (4, 4)
        Coefficients of the four cubic Lagrange basis polynomials.

        The i-th row contains the coefficients of the basis polynomial
        associated with the i-th interpolation node. The coefficients are
        ordered by descending powers:

            coeff[i] = [a3, a2, a1, a0],

        meaning that the i-th basis polynomial is

            a3 * x^3 + a2 * x^2 + a1 * x + a0.
    """
    scale = 3 * n

    t0 = left * scale
    t1 = t0 + 1
    t2 = t1 + 1
    t3 = t2 + 1

    return np.array(
        [
            -1 / 6 * np.array([scale**3, -(scale**2) * (t1 + t2 + t3), scale * (t1 * t2 + t1 * t3 + t2 * t3), -t1 * t2 * t3]),
            1 / 2 * np.array([scale**3, -(scale**2) * (t0 + t2 + t3), scale * (t0 * t2 + t0 * t3 + t2 * t3), -t0 * t2 * t3]),
            -1 / 2 * np.array([scale**3, -(scale**2) * (t0 + t1 + t3), scale * (t0 * t1 + t0 * t3 + t1 * t3), -t0 * t1 * t3]),
            1 / 6 * np.array([scale**3, -(scale**2) * (t0 + t1 + t2), scale * (t0 * t1 + t0 * t2 + t1 * t2), -t0 * t1 * t2]),
        ]
    )


def f0(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2) dx

    using the analytical formula based on the error function.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return np.sqrt(np.pi) / 2 * (erf(eta * b) - erf(eta * a)) / eta


def f1(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2)*x dx

    using the analytical formula.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return (np.exp(-eta * eta * a * a) - np.exp(-eta * eta * b * b)) / (2 * eta * eta)


def f2(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2)*x^2 dx

    using the analytical formula.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return a * np.exp(-eta * eta * a * a) / (2 * eta * eta) - b * np.exp(-eta * eta * b * b) / (2 * eta * eta) + f0(eta, a, b) / (2 * eta * eta)


def f3(eta, a, b):
    """
    Compute the integral

        ∫_a^b exp(-eta^2 x^2)*x^3 dx

    using the analytical formula.

    Parameters
    ----------
    eta : float
        Scaling parameter in the Gaussian kernel.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        The value of the integral.
    """
    return a * a * np.exp(-eta * eta * a * a) / (2 * eta * eta) - b * b * np.exp(-eta * eta * b * b) / (2 * eta * eta) + f1(eta, a, b) / (eta * eta)


def kernel_integral_0_to_1(delta, x):
    """
    Compute the integral

        ∫_0^1 R_delta(x, y) dy

    using f0.

    Parameters
    ----------
    delta : float
        Nonlocal interaction radius.
    x : float or np.ndarray

    Returns
    -------
    float
        The value of the integral.
    """
    return f0(1 / delta, -x, 1 - x) / delta


def build_quadrature(n, delta, num_leggauss=5):
    """
    Build Gauss-Legendre quadrature nodes and weights on the cell [0, 1 / n].

    If the cell is sufficiently longer than the nonlocal scale, namely
    1 / n > 6 * delta, the cell is split into three subintervals:

        [0, 3 * delta],
        [3 * delta, 1 / n - 3 * delta],
        [1 / n - 3 * delta, 1 / n].

    Gauss-Legendre quadrature with `num_leggauss` points is then applied
    on each subinterval.

    Otherwise, Gauss-Legendre quadrature with `3 * num_leggauss` points
    is applied directly on [0, 1 / n].

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    delta : float
        Nonlocal interaction radius.
    num_leggauss : int, optional
        Number of Gauss-Legendre points used on each subinterval.
        Default is 5.

    Returns
    -------
    quadrature_nodes : np.ndarray of shape (3 * num_leggauss,)
        Quadrature nodes on [0, 1 / n].
    quadrature_weights : np.ndarray of shape (3 * num_leggauss,)
        Quadrature weights corresponding to `quadrature_nodes`.
    """
    singular_width = 3 * delta
    if 1 / n > 2 * singular_width:
        leggauss_nodes, leggauss_weights = leggauss(num_leggauss)

        a = 0
        b = singular_width
        left_nodes = 0.5 * (b - a) * leggauss_nodes + 0.5 * (b + a)
        left_weights = 0.5 * (b - a) * leggauss_weights

        a = singular_width
        b = 1 / n - singular_width
        middle_nodes = 0.5 * (b - a) * leggauss_nodes + 0.5 * (b + a)
        middle_weights = 0.5 * (b - a) * leggauss_weights

        right_nodes = 1 / n - np.flip(left_nodes)
        right_weights = np.flip(left_weights)

        quadrature_nodes = np.concatenate([left_nodes, middle_nodes, right_nodes])
        quadrature_weights = np.concatenate([left_weights, middle_weights, right_weights])
    else:
        leggauss_nodes, leggauss_weights = leggauss(3 * num_leggauss)
        a = 0
        b = 1 / n
        quadrature_nodes = 0.5 * (b - a) * leggauss_nodes + 0.5 * (b + a)
        quadrature_weights = 0.5 * (b - a) * leggauss_weights

    return quadrature_nodes, quadrature_weights


def compute_relevant_cells_coeffs(n, quadrature_node, delta, tol=1e-12):
    """
    Compute kernel-weighted basis integrals for all relevant cells.
    For a fixed quadrature node x_quad in [0, 1 / n), this function computes

        ∫_{k/n}^{(k+1)/n} R_delta(x_quad, y) psi_i(y) dy,

    for integer cell offsets k = ..., -2, -1, 0, 1, 2, ... and
    i = 0, 1, 2, 3, where psi_i are the four cubic Lagrange basis
    polynomials on the cell [k / n, (k + 1) / n].

    For the computation on each single cell, see
    `compute_single_cell_coeffs`.

    Since the nonlocal interaction radius delta is typically small, the
    coefficients decay rapidly as |k| increases. The cell offsets are
    therefore truncated adaptively once the contributions on both sides
    are smaller than `tol`.

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    quadrature_node : float
        A single quadrature node in [0, 1 / n).
    delta : float
        Nonlocal interaction radius.
    tol : float, optional
        Truncation tolerance for ignoring far-away cell contributions.
        Default is 1e-12.

    Returns
    -------
    relevant_cells_coeffs : np.ndarray of shape (num_relevant_cells, 4)
        Kernel-weighted integrals over the relevant neighboring cells.
        The row `relevant_cells_coeffs[j]` corresponds to one cell offset (j - center_cell_index), and the
        four columns correspond to the four cubic basis functions on that cell.
    center_cell_index : int
        Index of the row in `relevant_cells_coeffs` corresponding to cell offset k = 0.
        In other words, the position of the cell that contains the quadrature node.
    """

    def compute_single_cell_coeffs(k):
        """
        Compute kernel-weighted basis integrals over a single cell.

        For a fixed quadrature node x_quad, this function evaluates

            I_i(k) = ∫_{k/n}^{(k+1)/n} R_delta(x_quad, y) psi_i(y) dy,

        for i = 0, 1, 2, 3, where R_delta is the scaled Gaussian kernel and
        psi_i are cubic Lagrange basis polynomials.

        The integral is computed analytically by combining the coefficients
        of each cubic basis polynomial with Gaussian moments.

        Parameters
        ----------
        k : int
            Cell index. The integration domain is [k / n, (k + 1) / n].

        Returns
        -------
        single_cell_coeffs : np.ndarray of shape (4,)
            Values [I_0(k), I_1(k), I_2(k), I_3(k)].
        """
        eta = 1 / delta

        # change of variable, simply shift
        left = k / n - quadrature_node
        right = (k + 1) / n - quadrature_node
        gaussian_moments = np.array([f3(eta, left, right), f2(eta, left, right), f1(eta, left, right), f0(eta, left, right)])
        basis_coeffs = cubic_lagrange_basis_coeffs(k / n - quadrature_node, n)
        single_cell_coeffs = gaussian_moments @ basis_coeffs.T / delta
        return single_cell_coeffs

    coeffs_by_cell = {0: compute_single_cell_coeffs(0)}

    cell_offset = 1
    while True:
        right_cell_impacts = compute_single_cell_coeffs(cell_offset)
        left_cell_impacts = compute_single_cell_coeffs(-cell_offset)

        if max(np.linalg.norm(right_cell_impacts), np.linalg.norm(left_cell_impacts)) < tol:
            break

        if np.linalg.norm(right_cell_impacts) >= tol:
            coeffs_by_cell[cell_offset] = right_cell_impacts

        if np.linalg.norm(left_cell_impacts) >= tol:
            coeffs_by_cell[-cell_offset] = left_cell_impacts

        cell_offset += 1

    cell_offsets = sorted(coeffs_by_cell.keys())
    relevant_cells_coeffs = np.array([coeffs_by_cell[k] for k in cell_offsets])
    center_cell_index = cell_offsets.index(0)

    return relevant_cells_coeffs, center_cell_index


def build_coeffs_one_row(n, relevant_cells_coeffs, center_cell_index, cell_idx):
    """
    Assemble the nonzero entries of the matrix row for one global quadrature node.

    For a fixed local quadrature node and a fixed `cell_idx`, this function
    accumulates the contributions from nearby relevant cells to this global
    quadrature node, and maps them to the corresponding global cubic basis nodes.

    The input `relevant_cells_coeffs` stores the kernel-weighted coefficients
    associated with relative cell offsets around the quadrature node. Near
    the boundary, only the valid subset of these offsets is active.

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    relevant_cells_coeffs : np.ndarray of shape (num_relevant_cells, 4)
        Kernel-weighted coefficients for relevant cell offsets. Each row
        contains the four contributions from one nearby cell.
    center_cell_index : int
        Index of the row in `relevant_cells_coeffs` corresponding to the
        cell offset k = 0, i.e. the cell containing the quadrature node.
    cell_idx : int
        Index of the cell containing the global quadrature node.

    Returns
    -------
    values : np.ndarray of shape (num_active_basis_nodes,)
        Nonzero values in the matrix row for this global quadrature node.
    cols : np.ndarray of shape (num_active_basis_nodes,)
        Column indices of the global cubic basis nodes corresponding to `values`.
    """

    offset_start = max(0, center_cell_index - cell_idx)
    offset_end = min(len(relevant_cells_coeffs), n - cell_idx + center_cell_index)
    num_active_cells = offset_end - offset_start
    num_active_basis_nodes = 3 * num_active_cells + 1

    values = np.zeros(num_active_basis_nodes)
    base_idx = 3 * np.arange(num_active_cells)[:, None] + np.arange(4)
    np.add.at(values, base_idx, relevant_cells_coeffs[offset_start:offset_end])

    col_start = 3 * max(0, cell_idx - center_cell_index)
    col_end = col_start + num_active_basis_nodes
    cols = np.arange(col_start, col_end, dtype=np.int64)
    return values, cols


def build_coeffs_matrix(n, quadrature_nodes, delta):
    """
    Compute the sparse matrix A such that

        ∫_0^1 R_delta(x,y) u(y) dy = A @ u_basis,

    where x denotes the global quadrature nodes of shape (n * num_quad, ),
    `u_basis` contains the values of u (to be solved) at the global cubic basis nodes of shape (3 * n + 1, ),
    A is the sparse matrix of shape (n * num_quad, 3 * n + 1).

    The matrix is assembled from `relevant_cells_coeffs` of shape (num_relevant_cells, 4),
    where each row contains the coefficients associated with one relevant cell offset.

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    quadrature_nodes : np.ndarray of shape (num_quad,)
        Quadrature nodes on [0, 1 / n).
    delta : float
        Nonlocal interaction radius.

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape (n * num_quad, 3 * n + 1)
        Sparse matrix whose rows correspond to global quadrature nodes and
        whose columns correspond to global cubic basis nodes.
    """
    row_blocks, col_blocks, data_blocks = [], [], []
    num_quad = len(quadrature_nodes)

    for quad_idx, quadrature_node in enumerate(quadrature_nodes):
        relevant_cells_coeffs, center_cell_index = compute_relevant_cells_coeffs(n, quadrature_node, delta)

        for cell_idx in range(n):
            global_coeffs, cols = build_coeffs_one_row(n, relevant_cells_coeffs, center_cell_index, cell_idx)
            rows = np.full(len(cols), quad_idx + cell_idx * num_quad, dtype=np.int64)

            row_blocks.append(rows)
            col_blocks.append(cols)
            data_blocks.append(global_coeffs)

    row_indices = np.concatenate(row_blocks)
    col_indices = np.concatenate(col_blocks)
    data_entries = np.concatenate(data_blocks)

    A = sp.coo_matrix((data_entries, (row_indices, col_indices)), shape=(n * num_quad, 3 * n + 1))
    return A.tocsr()


def build_boundary_coeffs_vector(n, delta):
    """
    Build boundary coefficient vectors at x = 0 and x = 1 such that

        ∫_0^1 K_delta(0,y) u(y) dy = A_bd_0 @ u_basis,
        ∫_0^1 K_delta(1,y) u(y) dy = A_bd_1 @ u_basis,

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    delta : float
        Nonlocal interaction radius.

    Returns
    -------
    A_bd_0, A_bd_1 : np.ndarray of shape (3 * n + 1,)
        Boundary coefficients corresponding to x = 0 and x = 1.
    """
    relevant_cells_coeffs, center_cell_index = compute_relevant_cells_coeffs(n, 0, delta)

    # x = 0
    global_coeffs, cols = build_coeffs_one_row(n, relevant_cells_coeffs, center_cell_index, 0)
    A_bd_0 = np.zeros(3 * n + 1)
    A_bd_0[cols] = global_coeffs

    # x = 1
    global_coeffs, cols = build_coeffs_one_row(n, relevant_cells_coeffs, center_cell_index, n)
    A_bd_1 = np.zeros(3 * n + 1)
    A_bd_1[cols] = global_coeffs

    return A_bd_0, A_bd_1


def build_interp_matrix(n, x_interp):
    """
    Compute the sparse matrix A such that

        u(x_interp)= A @ u_basis,

    where `x_interp` denotes points of shape (num_interp, ) to be interpolated using cubic basis polynomials,
    `u_basis` contains the values of u (to be solved) at the global cubic basis nodes of shape (3 * n + 1, ),
    A is the sparse matrix of shape (num_interp, 3 * n + 1).

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.
    x_interp : np.ndarray of shape (num_interp, )
        Points in [0, 1].

    Returns
    -------
    A : scipy.sparse.csr_matrix of shape (num_interp, 3 * n + 1)
        Sparse matrix whose rows correspond to x_interp and
        whose columns correspond to global cubic basis nodes.
    """
    num_interp = len(x_interp)

    # x_interp = q * (1 / n) + r
    q = np.floor(n * x_interp).astype(int)
    q = np.clip(q, 0, n - 1)  # corner case x_interp == 1
    r = x_interp - q / n

    interp_coeffs = np.stack([np.polyval(coeff, r) for coeff in cubic_lagrange_basis_coeffs(0, n)], axis=1).ravel()
    i, j = np.indices((num_interp, 4))
    row_indices = i.ravel()
    col_indices = (q[:, None] * 3 + j).ravel()
    A = sp.coo_matrix((interp_coeffs, (row_indices, col_indices)), shape=(num_interp, 3 * n + 1))

    return A.tocsr()


def build_simpson_weights(n):
    """
    Construct composite Simpson (3/8 rule) weights on a uniform cubic grid.

    The grid consists of (3n + 1) nodes corresponding to n cells with
    cubic Lagrange basis (3 degrees of freedom per cell plus one endpoint).

    Parameters
    ----------
    n : int
        Number of cells with cell size 1 / n.

    Returns
    -------
    simpson_weights : np.ndarray of shape (3 * n + 1,)
        Quadrature weights corresponding to the global cubic nodes.
    """
    simpson_weights = np.full(3 * n + 1, 3 / 8)

    # every third node
    simpson_weights[::3] = 1 / 4

    # endpoints
    simpson_weights[0] = 1 / 8
    simpson_weights[-1] = 1 / 8

    return simpson_weights / n
