""" This module implements Stiefel optimization as described
in https://proceedings.neurips.cc/paper/2016/file/d9ff90f4000eacd3a6c9cb27f78994cf-Paper.pdf
"""

import torch
from src.rnn import StiefelParameter


class StiefelOptimizer(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99,
                 eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super().__init__(params, lr, alpha, eps, weight_decay, momentum, centered)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('sparse gradients are not suppported')
                state = self.state[p]

                if type(p) == StiefelParameter:
                    # http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf
                    assert p.shape[0] == p.shape[1]
                    eye = torch.eye(p.shape[0])
                    A = torch.matmul(grad, p.t()) \
                        - torch.matmul(p, grad.t())
                    cayleyDenom = eye + (group['lr']/2.0) * A
                    cayleyNumer = eye - (group['lr']/2.0) * A
                    C = torch.matmul(torch.inverse(cayleyDenom), cayleyNumer)
                    # print('stop')
                    pnew = torch.matmul(C, p)
                    p.data = pnew
                    # print('stop')
                    
                else:
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['momentum'] > 0:
                            state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['centered']:
                            state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    square_avg = state['square_avg']
                    alpha = group['alpha']

                    state['step'] += 1

                    if group['weight_decay'] != 0:
                        grad = grad.add(p, alpha=group['weight_decay'])

                    square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                    if group['centered']:
                        grad_avg = state['grad_avg']
                        grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                        avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                    else:
                        avg = square_avg.sqrt().add_(group['eps'])

                    if group['momentum'] > 0:
                        buf = state['momentum_buffer']
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.add_(buf, alpha=-group['lr'])
                    else:
                        p.addcdiv_(grad, avg, value=-group['lr'])

        return loss
