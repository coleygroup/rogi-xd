from torch import Tensor

def reconstruction_accuracy(xs: list[Tensor], xs_rec: list[Tensor]) -> float:
    return sum(x.equal(x_rec) for x, x_rec in zip(xs, xs_rec)) / len(xs)
