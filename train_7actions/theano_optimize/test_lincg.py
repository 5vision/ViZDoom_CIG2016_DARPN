import numpy
import time
import theano
import theano.tensor as T
from scipy import linalg
floatX = theano.config.floatX

import lincg

rng = numpy.random.RandomState(23091)
nparams = 1000

def init_psd_mat(size):
    temp = rng.rand(size, size)
    return numpy.dot(temp.T, temp) + numpy.eye(size)*.001

symb = {}
symb['L'] = T.matrix("L")
symb['g'] = T.vector("g")
vals = {}
vals['L'] = init_psd_mat(nparams).astype(floatX)
vals['g'] = rng.rand(nparams).astype(floatX)

## now compute L^-1 g
vals['Linv_g_cho'] = linalg.cho_solve(linalg.cho_factor(vals['L']), vals['g'])
vals['Linv_g_sol'] = linalg.solve(vals['L'], vals['g'])

def test_lincg():
    rval = lincg.linear_cg(
            lambda x: [T.dot(symb['L'], x)],
            [symb['g']],
            rtol=1e-20,
            damp = 0.,
            maxit = 100000,
            floatX = floatX,
            profile=0)

    f = theano.function([symb['L'], symb['g']], rval[0])
    t1 = time.time()
    Linv_g = f(vals['L'], vals['g'])
    print 'test_lincg runtime (s):', time.time() - t1
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g_sol'], decimal=5)
    numpy.testing.assert_almost_equal(Linv_g, vals['Linv_g_cho'], decimal=5)


if __name__=='__main__':
    test_lincg()
