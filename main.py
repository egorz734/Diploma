import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import BSpline

def _coeff_of_devided_diff(x):
    '''
    Returns the coefficients of the devided difference.

    Parameters
    ----------
    x : array_like, shape (n,)
        Array which is used for the computation of devided
        difference.

    Returns
    -------
    res : array_like, shape (n,)
        Coefficients of the devided difference.
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

def _make_design_matrix(x, t):
    '''
    Returns a design matrix in the basis used by Woltring in
    the GCVSPL package.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.

    Returns
    -------
    Q : array_like, shape (n, n)
        Design matrix in the basis used by Woltring with the vector 
        of knots equal to ``x``.
    
    Notes
    -----
    In order to evaluate basis elements in the Woltring basis, we select
    the vector of knots ``t`` for the B-spline basis in a special way: 
    central knots are equal to the ``x``. We add 3 elements from both 
    left and right sides equal to ``x[0]`` and ``x[-1]`` accordingly.
    After that, we add up columns of the design matrix of B-splines in a
    way to move to the Basis used in GCVSPL.
    '''

    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')

    n = x.shape[0]
    nt = t.shape[0]
    X = BSpline.design_matrix(x, t, 3).toarray()
    Q = np.zeros((n, nt - 6))

    # 'central' basis elements are equal
    Q[:,2:-2] = X[:,3:-3]
    # equations for the first and last two elemts
    Q[:,0] = (t[5]+t[4]-2*t[3])*X[:,0] + (t[5]-t[3])*X[:,1]
    Q[:,1] = X[:,0] + X[:,1] + X[:,2]
    Q[:,-2] = X[:,-3] + X[:,-2] + X[:,-1]
    Q[:,-1] = (t[-4]-t[-6])*X[:,-2]+(2*t[-4]-t[-5]-t[-6])*X[:,-1]
    return Q

def make_smoothing_spline(x, y, w=None):
    '''
    Returns a smoothing spline function using the GCV criteria.

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
    func : callable
        A callable representing a spline in the Woltring basis 
        as a solution of the problem of smoothing splines using
        the GCV criteria.
    
    Notes
    -----
    GCV - generalized cross-validation.
    '''
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)

    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')

    if w is None:
        w = [1.] * len(x)
    w = np.ascontiguousarray(w)
    if all(w > 0):
        # inverse the matrix of weights
        w = np.diag(1. / w)
    else:
        raise ValueError('Invalid vector of weights')

    t = np.r_[[x[0]]*3,x,[x[-1]]*3]
    X = _make_design_matrix(x, t)

    n = len(x)
    E = np.zeros((n, n))
    E[:3,0] = _coeff_of_devided_diff(x[:3])
    E[:4,1] = _coeff_of_devided_diff(x[:4])
    for j in range(2, n - 2):
        E[j-2:j+3, j] = (x[j+2]-x[j-2])*_coeff_of_devided_diff(x[j-2:j+3])
    E[-4:, -2] = -_coeff_of_devided_diff(x[-4:])
    E[-3:, -1] =  _coeff_of_devided_diff(x[-3:])

    for i in range(n):
        E[i] *= 6. / w[i, i]

    p = _compute_optimal_gcv_parameter(X, E, y, w)
    func = _make_spline(X, E, t, y, p, w)
    return func
    
def _make_spline(X, E, t, y, p, w):
    '''
    Returns a spline function with regularization parameter equal
    to ``p`` using a matrix representation of the penalty function.

    Parameters
    ----------
    X : array_like, shape (n, n)
        Design matrix in the Woltring basis
    E : array_like, shape (n, n)
        Matrix representation of the penalty function
    y : array_like, shape (n,)
        Ordinates.
    p : float
        Regularization parameter (``p >= 0``)
    w : array_like, shape (n, n)
        An inverse for the matrix of weigts (diagonal matrix)

    Returns
    -------
    func : callable
        A callable representing a spline in the Woltring basis 
        with vector of coefficients ``c`` as a solution of the 
        regularized wighted least squares problem with inverse
        of the matrix of weights ``w`` and regularization parameter
        ``p``.
    '''
    # get a vector of coefficients from the weighted regularized OLS
    # with parameter p
    def eval_spline(x, c):
        x = np.ascontiguousarray(x)
        X = _make_design_matrix(x, t)
        return sum([X[:,i] * c[i] for i in range(len(c))])

    c = np.linalg.solve(X + p * w @ E, y)
    func = lambda x: eval_spline(x, c)
    return func
    
def _compute_optimal_gcv_parameter(X, E, y, w):
    '''
    Returns regularization parameter from the GCV criteria

    Parameters
    ----------
    X : array_like, shape (n, n)
        Design matrix in the Woltring basis
    E : array_like, shape (n, n)
        Matrix representation of the penalty function
    y : array_like, shape (n,)
        Ordinates.
    w : array_like, shape (n, n)
        An inverse for the matrix of weigts (diagonal matrix)

    Returns
    -------
    lam : float
        An optimal from the GCV criteria point of view regularization
        parameter.
    '''
    n = len(y)
    wE = w @ E
    def _gcv(lam):
        '''
        Computes the GCV criteria
        
        Parameters
        ----------
        lam : float
            Regularization parameter.
        
        Returns
        -------
        res : float
            Value of the GCV criteria with the regularization parameter
            ``lam``.
        '''
        A = X @ np.linalg.inv(X + lam * wE)
        tmp = np.eye(n)-A
        numerator = np.linalg.norm(tmp @ y)**2/n**2
        denom = (np.trace(tmp)/n)**2
        res = numerator/denom
        return res

    # Golden section method was selected by Woltring in the GCV package
    gcv_est = minimize_scalar(_gcv, bounds=(0,np.inf),
                                    method='Golden', tol=1e-15,
                                    options={'maxiter': 500})
    if gcv_est.success:
        return gcv_est.x
    raise ValueError(f"Unable to find minimum of the GCV "
                     f"function: {gcv_est.message}")