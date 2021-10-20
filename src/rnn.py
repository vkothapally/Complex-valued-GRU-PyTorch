import torch
from torch.nn.parameter import Parameter


class StiefelParameter(Parameter):
    pass




class ModRelu(torch.nn.Module):
    """
        Implementation of the modRelu from Arjovski et al.
        f(z) = relu(|z| + b)(z / |z|) or
        f(r,theta) = relu(r + b)e^(i*theta)
        b is initialized to zero, this leads to a network, which
        is linear during early optimization.

    Translated from:
    https://github.com/v0lta/
        Complex-gated-recurrent-neural-networks/blob/master/custom_cells.py

    Input:
        z: complex input.
        b: 'dead' zone radius.
    Returns:
        z_out: complex output.
    Reference:
         Arjovsky et al. Unitary Evolution Recurrent Neural Networks
         https://arxiv.org/abs/1511.06464
    """

    def __init__(self):
        super().__init__()
        self.b = Parameter(torch.rand([1])*0.01)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        modulus = torch.sqrt(z.real**2 + z.imag**2)
        rescale = torch.nn.functional.relu(modulus + self.b) / \
            (modulus + 1e-8)
        rescale = torch.complex(rescale, torch.zeros_like(rescale))
        return rescale*z



class SimpleRecurrentCell(torch.nn.Module):
    """ A simple Recurrent network cell. """
    def __init__(self, hidden_size: int = 250,
                 input_size: int=1,
                 activation: torch.nn.Module=torch.nn.Tanh):
        super().__init__()
        self.hidden_size = hidden_size
        # generate an orthogonal initialization.
        U, _, V = torch.svd(torch.randn(hidden_size, hidden_size))
        self.weight_recurrent = StiefelParameter(torch.matmul(U, V))
        self.weight_input = Parameter(torch.randn(input_size, hidden_size))
        self.bias_input = Parameter(torch.zeros(hidden_size))

        self.activation = activation()

    def forward(self, in_tensor, hidden_tensor):
        hidden_new = torch.matmul(hidden_tensor, self.weight_recurrent)
        in_new = torch.matmul(in_tensor, self.weight_input)
        activation_new = self.activation(hidden_new + in_new + self.bias_input)
        return activation_new

    def zero_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class ComplexCell(torch.nn.Module):
    """ A complex Recurrent network cell. """
    def __init__(self, hidden_size: int = 250,
                 input_size: int=1,
                 activation: torch.nn.Module=ModRelu):
        super().__init__()
        self.hidden_size = hidden_size
        # generate an orthogonal initialization.
        U, _, V = torch.svd(
            torch.complex(torch.randn(hidden_size, hidden_size),
                          torch.randn(hidden_size, hidden_size)))
        self.weight_recurrent = StiefelParameter(torch.matmul(U, V))
        print("init norm: {:2.2f}".format(torch.linalg.norm(self.weight_recurrent, 2).item()))
        self.weight_input = Parameter(torch.randn(input_size, hidden_size)).to(torch.cfloat)
        self.bias_input = Parameter(torch.zeros(hidden_size)).to(torch.cfloat)

        self.activation = activation()

    def forward(self, in_tensor, hidden_tensor):
        if in_tensor.dtype == torch.float:
            in_tensor = in_tensor.to(torch.cfloat)
        hidden_new = torch.matmul(hidden_tensor, self.weight_recurrent)
        in_new = torch.matmul(in_tensor, self.weight_input)
        activation_new = self.activation(hidden_new + in_new + self.bias_input)
        return activation_new

    def zero_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(torch.cfloat)



class RecurrentLayer(torch.nn.Module):
    def __init__(self, cell, output_size=1):
        super().__init__()
        self.cell = cell

        if type(self.cell) is ComplexCell:
            self.weight_output = Parameter(torch.randn(cell.hidden_size*2, output_size))
        else:   
            self.weight_output = Parameter(torch.randn(cell.hidden_size, output_size))

        self.bias_output = Parameter(torch.randn(output_size))


    def forward(self, input_series, steps, init_state=None):
        if init_state is None:
            batch_size = input_series.shape[0]
            init_state = self.cell.zero_state(batch_size)
        
        output_lst = []
        hidden_state = init_state
        for step in range(steps):
            hidden_state = self.cell(input_series[:, step, :],
                                     hidden_state)
            if type(self.cell) is ComplexCell:
                cat_state = torch.cat([hidden_state.real, hidden_state.imag], -1)
            else:
                cat_state = hidden_state
            output = torch.matmul(cat_state, self.weight_output)
            output = output  + self.bias_output
            output_lst.append(output)
        return torch.stack(output_lst, dim=1)

    def zero_state(self, batch_size: int):
        return self.cell.zero_state(batch_size)