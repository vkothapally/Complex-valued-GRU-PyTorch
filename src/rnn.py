import torch
from torch.nn.parameter import Parameter


class StiefelParameter(Parameter):
    pass




def mod_relu(z, scope='', reuse=None):
    """
        Implementation of the modRelu from Arjovski et al.
        f(z) = relu(|z| + b)(z / |z|) or
        f(r,theta) = relu(r + b)e^(i*theta)
        b is initialized to zero, this leads to a network, which
        is linear during early optimization.
    Input:
        z: complex input.
        b: 'dead' zone radius.
    Returns:
        z_out: complex output.
    Reference:
         Arjovsky et al. Unitary Evolution Recurrent Neural Networks
         https://arxiv.org/abs/1511.06464
    """
    # b = tf.get_variable('b', [], dtype=tf.float32,
    #                    initializer=urnd_init(-0.01, 0.01))
    # modulus = tf.sqrt(tf.real(z)**2 + tf.imag(z)**2)
    # rescale = tf.nn.relu(modulus + b) / (modulus + 1e-6)
    # rescale = tf.complex(rescale, tf.zeros_like(rescale))
    # return tf.multiply(rescale, z)
    # TODO: finish translating this code to PyTorch.
    pass



class SimpleRecurrentCell(torch.nn.Module):
    """ A simple Recurrent network cell. """
    def __init__(self, hidden_size=250, input_size=1,
                 activation=torch.nn.Tanh):
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
    # TODO.
    pass


class RecurrentLayer(torch.nn.Module):
    def __init__(self, cell, output_size=1):
        super().__init__()
        self.cell = cell
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
            output = torch.matmul(hidden_state, self.weight_output)
            output = output  + self.bias_output
            output_lst.append(output)
        return torch.stack(output_lst, dim=1)