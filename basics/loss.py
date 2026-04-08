import torch
import torch.nn as nn


def cross_entropy(inputs, targets):

    vocab_size = inputs.shape[-1]
    flat_inputs = inputs.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)

    max_val = torch.max(flat_inputs, dim=-1, keepdim=True).values
    shifted_inputs = flat_inputs - max_val 

    target_shifted_logits = shifted_inputs[torch.arange(flat_targets.shape[0]), flat_targets]

    lse_shifted = torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1))

    losses = lse_shifted - target_shifted_logits

    return losses.mean()





