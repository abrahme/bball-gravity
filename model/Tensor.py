import torch.nn as nn
import numpy as np
import tensorly as tl
from tensorly.tenalg import inner


class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(input_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)

        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)

        self.n_outputs = int(np.prod(output_size[1:]))

        # core of regression tensor weights
        self.core = nn.Parameter(
            tl.zeros(self.ranks), requires_grad=True
        )

        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)

        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])

        ## add and register factors
        self.factors = []

        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter(f'factor_{index}', self.factors[index])

        ### init params
        self.core.data.uniform_(-.1, .1)
        for f in self.factors:
            f.data.uniform_(-.1, .1)

    def forward(self, x):
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        return inner(x, regression_weights, n_modes=tl.ndim(x) - 1) + self.bias

    def penalty(self, order=2):
        penalty = tl.norm(self.core, order)
        for f in self.factors:
            penalty = penalty + tl.norm(f, order)
        return penalty
