import numpy as np
import torch


def pair_wise_eud(x,y,c=1.0):
    # input: 
    # x, m x d
    # y, n x d
    # output: m x n
    m = x.size(0)
    n = y.size(0)
    d = x.size(1)
    assert(x.size(1) == y.size(1))
    xx = x.pow(2).sum(-1, keepdim = True)
    yy = y.pow(2).sum(-1, keepdim = True)
    xy = torch.einsum('ij,kj->ik', (x, y))

    result = xx - 2*xy + yy.permute(1, 0)
    return result

def pair_wise_cos(x,y,c=1.0):
    # input: 
    # x, m x d
    # y, n x d
    # output: m x n
    x_norm = torch.norm(x, dim=1, keepdim = True)# m x 1
    y_norm = torch.norm(y, dim=1, keepdim = True)# n x 1

    denominator = torch.matmul(x_norm, y_norm.t())  # m x n
    numerator = torch.matmul(x, y.t())  # m x n, each element is a in-prod
    
    return -numerator / denominator  # m x n

def pair_wise_hyp(x, y, c=1.0):
    c = torch.as_tensor(c)
    return _dist_matrix(x, y, c)

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def tensor_dot(x, y):
    res = torch.einsum('ij,kj->ik', (x, y))
    return res

def _mobius_addition_batch(x, y, c):
    xy = tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = (1 + 2 * c * xy + c * y2.permute(1, 0))  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1) # B x C x 1 * B x 1 x D = B x C x D
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D + B x 1 x 1 = B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0) # B x 1 * 1 x C = B x C
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res

def _mobius_addition_same_size(x, y, c):
    xy = torch.einsum('ij,ij -> i', (x, y)).unsqueeze(1) # n x1
    x2 = x.pow(2).sum(-1, keepdim=True)  # n x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # n x 1
    num = 1 + 2 * c * xy + c * y2  # n x 1
    num2 = num * x # n x D
    num3 = num + (1 - c * x2) * y  # n x D
    denom_part1 = 1 + 2 * c * xy  # n x 1
    denom_part2 = c ** 2 * x2 * y2 # n x 1
    denom = denom_part1 + denom_part2
    res = num3 / (denom + 1e-5)
    
    return res

def _dist_matrix(x, y, c):
    sqrt_c = c ** 0.5
    return 2 / sqrt_c * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))


def dist_same_size(x,y, c=1.0):
    c = torch.as_tensor(c)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c * artanh(sqrt_c * torch.norm(_mobius_addition_same_size(-x, y, c=c), dim=-1))

def project(x, *, c=1.0):
    r"""
    Safe projection on the manifold for numerical stability. This was mentioned in [1]_
    Parameters
    """
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)

def _project(x, c):
    """Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        projected vector on the manifold"""
    c = torch.as_tensor(c).type_as(x)

    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def mobius_matvec(m, x, c):
    x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    sqrt_c = c ** 0.5
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return _project(res, c)

def euc2hyp(x, c):
    new_x = _project(expmap0(x, c), c)
    return new_x

### 以下为原始代码
def dist(x, y, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball
    .. math::
        d_c(x, y) = \frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|(-x)\oplus_c y\|_2)
    .. plot:: plots/extended/poincare/distance.py
    Parameters
    ----------
    x : tensor
        point on poincare ball
    y : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c, keepdim=keepdim)


def _dist(x, y, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c

def mobius_add(x, y, *, c=1.0):
    r"""
    Mobius addition is a special operation in a hyperbolic space.
    .. math::
        x \oplus_c y = \frac{
            (1 + 2 c \langle x, y\rangle + c \|y\|^2_2) x + (1 - c \|x\|_2^2) y
            }{
            1 + 2 c \langle x, y\rangle + c^2 \|x\|^2_2 \|y\|^2_2
        }
    In general this operation is not commutative:
    .. math::
        x \oplus_c y \ne y \oplus_c x
    But in some cases this property holds:
    * zero vector case
    .. math::
        \mathbf{0} \oplus_c x = x \oplus_c \mathbf{0}
    * zero negative curvature case that is same as Euclidean addition
    .. math::
        x \oplus_0 y = y \oplus_0 x
    Another usefull property is so called left-cancellation law:
    .. math::
        (-x) \oplus_c (x \oplus_c y) = y
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    y : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        the result of mobius addition
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)


def _mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)


def expmap0(u, *, c=1.0):
    r"""
    Exponential map for Poincare ball model from :math:`0`.
    .. math::
        \operatorname{Exp}^c_0(u) = \tanh(\sqrt{c}/2 \|u\|_2) \frac{u}{\sqrt{c}\|u\|_2}
    Parameters
    ----------
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c)


def _expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def _hyperbolic_softmax(X, A, P, c):
    lambda_pkc = 2 / (1 - c * P.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(c)
    mob_add = _mobius_addition_batch(-P, X, c)
    num = 2 * torch.sqrt(c) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)
    denom = torch.norm(A, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))
    logit = k.unsqueeze(1) * arsinh(num / denom)
    return logit.permute(1, 0)

class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(1e-5).log_()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
    
def artanh(x):
    return Artanh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)
