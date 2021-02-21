import torch
from torch.optim import SGD


class EmGD(SGD):

    def __init__(self, params, lr, momentum=0, m=1, k=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if m < 0.0:
            raise ValueError("Invalid anxiety value: {}".format(m))
        if k < 0.0:
            raise ValueError("Invalid confidence value: {}".format(k))

        defaults = dict(lr=lr, momentum=momentum, m=m, k=k)

        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            m = group['m']
            k = group['k']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                d_p_cog = d_p[:, :-1] if len(d_p.shape) > 1 else d_p[:-1]
                d_p_ven = d_p[:, -1:] if len(d_p.shape) > 1 else d_p[-1:]

                # Apply learning rate
                d_p_cog.mul_(lr)
                d_p_ven.mul_(m)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf_cog = buf[:, :-1] if len(buf.shape) > 1 else buf[:-1]
                        buf_ven = buf[:, -1:] if len(buf.shape) > 1 else buf[-1:]

                        buf_cog.mul_(momentum).add_(d_p_cog)
                        buf_ven.mul_(k).add_(d_p_ven)
                    else:
                        buf = param_state['momentum_buffer']

                        buf_cog = buf[:, :-1] if len(buf.shape) > 1 else buf[:-1]
                        buf_ven = buf[:, -1:] if len(buf.shape) > 1 else buf[-1:]
                        buf_cog.mul_(momentum).add_(1, d_p_cog)
                        buf_ven.mul_(k).add_(1, d_p_ven)

                    dim = 1 if len(buf.shape) > 1 else 0
                    d_p = torch.cat((buf_cog, buf_ven), dim=dim)

                p.data.add_(-1, d_p)

        return loss

    if '__name__' == '__main__':
        pass