from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                # Collect all parameters with gradients in this group
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                # Compute global norm
                torch.nn.utils.clip_grad_norm_(params_with_grad, group['max_grad_norm'])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction (efficient version from Algorithm 2 in the paper)
                # Instead of computing m_hat and v_hat, we compute corrected step size
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = alpha / bias_correction1
                    bias_correction2_sqrt = torch.sqrt(torch.tensor(bias_correction2))
                else:
                    step_size = alpha
                    bias_correction2_sqrt = 1.0

                # Update parameters
                # theta_t = theta_{t-1} - step_size * m_t / (sqrt(v_t) / sqrt(bias_correction2) + eps)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Add weight decay after the main gradient-based updates.
                # The learning rate is incorporated into this update.
                if group['weight_decay'] > 0.0:
                    p.data.add_(p.data, alpha=-alpha * group['weight_decay'])

        return loss