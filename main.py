import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import BSpline
from scipy.linalg import solve_banded

def _coeff_of_divided_diff(x):
    '''
    Returns the coefficients of the divided difference.

    Parameters
    ----------
    x : array_like, shape (n,)
        Array which is used for the computation of divided
        difference.

    Returns
    -------
    res : array_like, shape (n,)
        Coefficients of the divided difference.
    '''
    n = x.shape[0]
    res = np.zeros(n)
    for i in range(n):
        pp = 1.
        for k in range(n):
            if k != i:
                pp *= 1. / (x[i] - x[k])
        res[i] = pp
    return res

def _prepare_input(x, y=None, w=None):
    '''
    Does checks and prepares for the input of components 
    of the ``make_smoothing_spline`` algorithm.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas (sorted).
    y : array_like, shape (n,), optional
        Ordinates.
    w : array_like, shape (n,), optional
        Vector of weights.

    Returns
    -------
    x : array, shape (n,)
        Checked and prepared abscissas.
    y : array, shape (n,), in case y is not None
        Checked and prepared ordinates.
    w : array, shape (n,)
        Checked and prepared vector of weights
    t : array, shape (n + 6,)
        Vector of knots

    Notes
    -----
    Vector of knots constructed from ``x`` in the following way:
    ``t = [x_1, x_1, x_1, x_1, x_2, ..., x_{n-1}, x_n, x_n, x_n,
    x_n]``.

    In case ``w`` is None, array of ``1.`` is returned.
    '''
    x = np.ascontiguousarray(x, dtype=float)

    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')

    if w is None:
        w = [1.] * len(x)
    w = np.ascontiguousarray(w)
    if any(w <= 0):
        raise ValueError('Invalid vector of weights')

    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]

    if y is None:
        return x, w, t

    y = np.ascontiguousarray(y, dtype=float)
    return x, y, w, t

def _make_design_matrix(x):
    '''
    Returns a design matrix in the basis used by Woltring in
    the GCVSPL package [1].

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.

    Returns
    -------
    Q : array_like, shape (5, n)
        Design matrix in the basis used by Woltring with the vector 
        of knots equal to ``x``. Matrix is stored in the diagonal way.
    
    Notes
    -----
    The matrix is obtained from the design matrix in the B-splines basis
    using equalities (2.1.7).

    The matrix ``Q`` is tridiagonal and stored in the diagonal way (see
    scipy.linalg.solve_banded). As far as the penalty matrix contains 5
    diagonals, we add the first and the last rows of zeros to easily pass
    then the sum of these two matrices to solve_banded.

    References
    ----------
    [1] H. J. Woltring, “A Fortran package for generalized, cross-validatory
    spline smoothing and differentiation,” Advances in Engineering Software,
    vol. 8, no. 2, pp. 104–113, 1986.
    '''

    x, _, t = _prepare_input(x)

    n = x.shape[0]
    X = BSpline.design_matrix(x, t, 3)

    # central elements that do not change using sparse matrix
    ab = np.zeros((5, n))
    for i in range(1, 4):
        ab[i, 2:-2] = X[i: -4 + i, 3:-3][np.diag_indices(n - 4)]
    # changed basis elements stored in the banded way for
    # solve_banded

    # first elements
    ab[1, 1] = X[0, 0]
    ab[2, :2] = ((x[2] + x[1] - 2*x[0]) * X[0, 0], X[1, 1] + X[1, 2])
    ab[3, :2] = ((x[2] - x[0]) * X[1, 1], X[2, 2])

    # last elements
    ab[1, -2:] = (X[-3, -3], (x[-1] - x[-3]) * X[-2, -2])
    ab[2, -2:] = (X[-2, -3] + X[-2, -2], (2 * x[-1] - x[-2] - x[-3]) * X[-1, -1])
    ab[3, -2] = X[-1, -1]

    return ab

def _make_penalty_matrix(x, w=None):
    '''
    Returns a penalty matrix for the generalized cross-validation
    smoothing spline.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    w : array_like, shape (n,), optional
        Vector of weights.
    
    Returns
    -------
    E : array_like, shape (5, n)
        Penalty matrix. Matrix is stored in the diagonal way.
    
    Notes
    -----
    The penalty matrix is built from the coefficients of the divided
    differences using formulas discussed in section 1.4.1 (equation
    (1.4.5)).
    '''
    x, w, _ = _prepare_input(x, None, w)

    n = x.shape[0]
    ab = np.zeros((5, n))
    ab[2:, 0] = _coeff_of_divided_diff(x[:3]) / w[:3]
    ab[1:, 1] = _coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        ab[:, j] = (x[j+2]-x[j-2])*_coeff_of_divided_diff(x[j-2:j+3]) / w[j-2: j+3]

    ab[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    ab[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    ab *= 6

    return ab

def make_smoothing_spline(x, y, w=None):
    '''
    Returns a smoothing cubic spline function using the GCV criteria.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n,)
        Ordinates.
    w : array_like, shape (n,), optional
        Vector of weights.
    
    Returns
    -------
    func : a BSpline object.
        A callable representing a spline in the B-spline basis 
        as a solution of the problem of smoothing splines using
        the GCV criteria.
    
    Notes
    -----
    GCV - generalized cross-validation.
    '''
    x, y, w, _ = _prepare_input(x, y, w)    

    p = _compute_optimal_gcv_parameter(x, y, w)
    func = _make_spline(x, y, p, w)
    return func
    
def _make_spline(x, y, p, w=None):
    '''
    Returns a spline function with regularization parameter equal
    to ``p`` using a matrix representation of the penalty function.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n,)
        Ordinates.
    p : float
        Regularization parameter (``p >= 0``)
    w : array_like, shape (n,), optional
        Vector of weights.

    Returns
    -------
    func : a BSpline object.
        A callable representing a spline in the B-spline basis 
        as a solution of the problem of smoothing splines using
        the GCV criteria.

    Notes
    -----
    If ``p`` is 0 then the returns is equal to the interpolation
    spline with natural boundary conditions.
    '''
    # get a vector of coefficients from the weighted regularized OLS
    # with parameter p
    x, y, w, t = _prepare_input(x, y, w)
    
    if p < 0.:
        raise ValueError('Regularization parameter should be non-negative')

    X = _make_design_matrix(x)
    E = _make_penalty_matrix(x, w)

    c = solve_banded((2, 2), X + p * E, y)
    c_ = np.r_[ c[0] * (t[5] + t[4] - 2 * t[3]) + c[1],
                c[0] * (t[5] - t[3]) + c[1],
                c[1:-1],
                c[-1] * (t[-4] - t[-6]) + c[-2],
                c[-1] * (2 * t[-4] - t[-5] - t[-6]) + c[-2]]
    return BSpline.construct_fast(t, c_, 3)
    
def _compute_optimal_gcv_parameter(x, y, w=None):
    '''
    Returns regularization parameter from the GCV criteria

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n,)
        Ordinates.
    w : array_like, shape (n,), optional
        Vector of weights.

    Returns
    -------
    lam : float
        An optimal from the GCV criteria point of view regularization
        parameter.
    '''

    x, y, w, _ = _prepare_input(x, y, w)
    
    X = _make_design_matrix(x)
    E = _make_penalty_matrix(x, w)

    def _make_full_matrix(H, n):
        H_full = np.zeros((n, n), dtype=float)
        H_full[2:, :-2][np.diag_indices(n - 2)] = H[4, :-2]
        H_full[1:, :-1][np.diag_indices(n - 1)] = H[3, :-1]
        H_full[np.diag_indices(n)] = H[2]
        H_full[:-1, 1:][np.diag_indices(n - 1)] = H[1, 1:]
        H_full[:-2, 2:][np.diag_indices(n - 2)] = H[0, 2:]
        return H_full
    
    n = len(y)
    def _gcv(p):
        '''
        Computes the GCV criteria
        
        Parameters
        ----------
        p : float, (``p >= 0``)
            Regularization parameter.
        
        Returns
        -------
        res : float
            Value of the GCV criteria with the regularization parameter
            ``p``.
        
        Notes
        -----
        Criteria is computed from the following formula (1.3.2):
        $GCV(p) = \dfrac{1}{n} \sum\limits_{k = 1}^{n} \dfrac{ \left(
        y_k - f_{\lambda}(x_k) \right)^2}{\left( 1 - \Tr{A}/n\right)^2}$.
        The criteria is discussed in section 1.3.
        '''
        # Compute the denominator
        H = X + p * E

        # Transform matrix from banded storage to 2-D array
        H_full = _make_full_matrix(H, n)
        X_full = _make_full_matrix(X, n)

        H_inv = np.linalg.inv(H_full)
        A = X_full @ H_inv

        numer = np.linalg.norm((np.eye(n) - A) @ y)**2
        denom = (np.trace(np.eye(n) - A) / n)**2
    
        res = numer / denom
        
        return res

    # Golden section method was selected by Woltring in the GCV package
    gcv_est = minimize_scalar(_gcv, bracket=(1e-10, 2.),
                                    method='Golden', tol=1e-15,
                                    options={'maxiter': 500})
    if gcv_est.success:
        return gcv_est.x
    raise ValueError(f"Unable to find minimum of the GCV "
                     f"function: {gcv_est.message}")