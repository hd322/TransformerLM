import torch
import torch.nn as nn
import torch.optim as optim
import math

class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # 提取这组参数的超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # 更新时间步 t
                state['step'] += 1
                t = state['step']

                # 1. 应用 Weight Decay (Algorithm 1: Line 8)
                # \theta <- \theta - \alpha * \lambda * \theta
                # 提取公因式即: \theta <- \theta * (1 - lr * weight_decay)
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 2. 更新一阶动量估计 (Algorithm 1: Line 9)
                # m <- \beta_1 * m + (1 - \beta_1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. 更新二阶动量估计 (Algorithm 1: Line 10)
                # v <- \beta_2 * v + (1 - \beta_2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. 计算偏差校正后的 \alpha_t (Algorithm 1: Line 7)
                # \alpha_t = \alpha * \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # 5. 应用动量调整后的权重更新 (Algorithm 1: Line 11)
                # \theta <- \theta - \alpha_t * m / (\sqrt{v} + \epsilon)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    
def cosine_schedule(t, max_lr, min_lr, tw, tc):
    lr = min_lr
    if t < tw:
        lr = (t / tw) * max_lr
    elif tw <= t <= tc:
        freq = (t-tw) / (tc - tw)
        lr += (1 + math.cos(freq * math.pi)) * (max_lr - min_lr) / 2.0
        
    return lr
    