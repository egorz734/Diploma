import numpy as np
from scipy.interpolate import (make_interp_spline,
                               BSpline)
from numpy.testing import assert_allclose
from main import (make_smoothing_spline,
                  _make_design_matrix,
                  _make_penalty_matrix,
                  _make_spline)

class TestMakeSmoothingSpline:

    def test_compare_with_GCVSPL(self):
        '''
        Data is generated in the following way:
        >>> np.random.seed(1234)
        >>> n = 100
        >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
        >>> y = np.sin(x) + np.random.normal(scale=.5, size=n)

        Notes
        -----
        We obtain the result of performing the GCV smoothing splines
        package (by GCVSPL, Woltring) on the sample data points 
        using its version for Octave (https://github.com/srkuberski/gcvspl).
        In Octave we load up ``x`` and ``y`` and then perform the following
        script:
        >>> c = gcvsplmex( x, y, 2 );
        >>> y0 = spldermex( x, c, 2, x, 0 );
        '''
        # load the data sample
        data = np.load('data.npz')
        # data points
        x = data['x']
        y = data['y']
        
        y_GCVSPL = data['y_GCVSPL']
        y_compr = make_smoothing_spline(x, y)(x)
        assert_allclose(y_compr, y_GCVSPL, atol=1e-15, rtol=1e-6)
    
    def test_ridge_case(self):
        '''
        In case the regularization parameter is 0, the resulting spline
        is an interpolation spline with natural boundary conditions.
        '''
        # load the data sample
        data = np.load('data.npz')
        # data points
        x = data['x']
        y = data['y']
        n = x.shape[0]
        t = np.r_[[x[0]]*3, x, [x[-1]]*3]
        # in this case the matrix of weights does not matter
        # because it is multiplied by ``p`` which is 0.
        spline_GCV = _make_spline(x, y, 0.)
        spline_interp = make_interp_spline(x, y, 3, bc_type='natural')

        grid = np.linspace(x[0], x[-1], 2 * n)
        assert_allclose(spline_GCV(grid),
                        spline_interp(grid),
                        atol=1e-15)
    # def test_compare_with_weightedGCVSPL
    # no weights in github repo provided
