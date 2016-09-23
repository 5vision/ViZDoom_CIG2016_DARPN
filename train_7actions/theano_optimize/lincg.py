import theano
from theano import tensor
from theano.ifelse import ifelse
from theano.scan_module import scan

def linear_cg(compute_Gv, bs, rtol = 1e-6, maxit = 1000, damp=0, floatX = None, profile=0):
    """
    assume all are lists all the time
    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(bs)
    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2*n_params]
        xs = args[2*n_params:]
        _Aps = compute_Gv(*ps)[0]
        Aps = [x + damp*y for x,y in zip(_Aps, ps)]
        alpha = rsold/sum( (x*y).sum() for x,y in zip(Aps, ps))
        xs = [x + alpha * p for x,p in zip(xs,ps)]
        rs = [r - alpha * Ap for r, Ap, in zip(rs, Aps)]
        rsnew = sum( (r*r).sum() for r in rs)
        ps = [ r + rsnew/rsold*p for r,p in zip(rs,ps)]
        return [rsnew]+ps+rs+xs, \
                theano.scan_module.until(abs(rsnew) < rtol)

    r0s = bs
    _p0s = [tensor.unbroadcast(tensor.shape_padleft(x),0) for x in r0s]
    _r0s = [tensor.unbroadcast(tensor.shape_padleft(x),0) for x in r0s]
    _x0s = [tensor.unbroadcast(tensor.shape_padleft(
        tensor.zeros_like(x)),0) for x in bs]
    _rsold = sum( (r*r).sum() for r in r0s)
    #_rsold = tensor.unbroadcast(tensor.shape_padleft(rsold),0)
    outs, updates = scan(loop,
                         outputs_info = [_rsold] + _p0s + _r0s + _x0s,
                         n_steps = maxit,
                         mode = theano.Mode(linker='cvm'),
                         name = 'linear_conjugate_gradient',
                         profile=profile)
    fxs = outs[1+2*n_params:]
    #return [x[0] for x in fxs]
    # 5vision hacks
    x = theano.gradient.disconnected_grad(fxs[0][-1].flatten())
    residual = outs[0][-1]
    return [x, residual]

def linear_cg_precond(compute_Gv, bs, Msz, rtol = 1e-16, maxit = 100000, floatX = None):
    """
    assume all are lists all the time
    Reference:
        http://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    n_params = len(bs)
    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2*n_params]
        xs = args[2*n_params:]
        Aps = compute_Gv(*ps)
        alpha = rsold/sum( (x*y).sum() for x,y in zip(Aps, ps))
        xs = [x + alpha * p for x,p in zip(xs,ps)]
        rs = [r - alpha * Ap for r, Ap, in zip(rs, Aps)]
        zs = [ r/z for r,z in zip(rs, Msz)]
        rsnew = sum( (r*z).sum() for r,z in zip(rs,zs))
        ps = [ z + rsnew/rsold*p for z,p in zip(zs,ps)]
        return [rsnew]+ps+rs+xs, 
    theano.scan_module.until(abs(rsnew) < rtol)

    r0s = bs
    _p0s = [tensor.unbroadcast(tensor.shape_padleft(x/z),0) for x,z in zip(r0s, Msz)]
    _r0s = [tensor.unbroadcast(tensor.shape_padleft(x),0) for x in r0s]
    _x0s = [tensor.unbroadcast(tensor.shape_padleft(
        tensor.zeros_like(x)),0) for x in bs]
    rsold = sum( (r*r/z).sum() for r,z in zip(r0s, Msz))
    _rsold = tensor.unbroadcast(tensor.shape_padleft(rsold),0)
    outs, updates = scan(loop,
                         states = [_rsold] + _p0s + _r0s + _x0s,
                         n_steps = maxit,
                         mode = theano.Mode(linker='c|py'),
                         name = 'linear_conjugate_gradient',
                         profile=0)
    fxs = outs[1+2*n_params:]
    return [x[0] for x in fxs]


