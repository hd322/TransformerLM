from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

import regex as re
from collections import defaultdict, Counter

from basics.tokenizer import Tokenizer
from basics.utils import Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax, MultiHeadAttention, TransformerBlock, save_checkpoint, load_checkpoint
from basics.loss import cross_entropy
from basics.optimizer import AdamW, cosine_schedule

from einops import rearrange, einsum
import numpy as np

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    model = Linear(d_in, d_out)
    model.load_state_dict({"weight": weights})
    return model(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    vocab = Embedding(vocab_size, d_model)
    vocab.load_state_dict({"weight": weights})

    return vocab(token_ids)



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    swiglu = SwiGLU(d_model, d_ff)
    swiglu.w1.weight.copy_(w1_weight)
    swiglu.w2.weight.copy_(w2_weight)
    swiglu.w3.weight.copy_(w3_weight)

    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    d_k = Q.size(-1)

    QK = Q @ K.transpose(-1, -2) / (d_k ** 0.5)

    if mask is not None:
        QK = QK.masked_fill(mask == False, float("-inf"))

    A = softmax(QK, dim=-1)

    return A @ V

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiHeadAttention(d_model, num_heads)
    mha.query.weight.copy_(q_proj_weight)
    mha.key.weight.copy_(k_proj_weight)
    mha.value.weight.copy_(v_proj_weight)
    mha.wo.weight.copy_(o_proj_weight)
    return mha(in_features)



def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiHeadAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)
    mha.query.weight.copy_(q_proj_weight)
    mha.key.weight.copy_(k_proj_weight)
    mha.value.weight.copy_(v_proj_weight)
    mha.wo.weight.copy_(o_proj_weight)
    
    return mha(in_features, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RoPE(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    device = in_features.device
    dtype = in_features.dtype

    block = TransformerBlock(
        embed_dim=d_model,
        num_heads=num_heads,
        ffn_dim=d_ff,
        theta=theta,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype
    )

    with torch.no_grad():
        block.mha.query.weight.copy_(weights["attn.q_proj.weight"])
        block.mha.key.weight.copy_(weights["attn.k_proj.weight"])
        block.mha.value.weight.copy_(weights["attn.v_proj.weight"])
        block.mha.wo.weight.copy_(weights["attn.output_proj.weight"])
        
        block.ffn.w1.weight.copy_(weights["ffn.w1.weight"])
        block.ffn.w2.weight.copy_(weights["ffn.w2.weight"])
        block.ffn.w3.weight.copy_(weights["ffn.w3.weight"])
        
        block.ln1.weight.copy_(weights["ln1.weight"])
        block.ln2.weight.copy_(weights["ln2.weight"])

    block.eval()
    with torch.no_grad():
        seq_len = in_features.shape[1]
        token_positions = torch.arange(seq_len, device=device)
        output = block(in_features, token_positions=token_positions)

    return output



def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    device = in_indices.device
    dtype = weights["token_embeddings.weight"].dtype

    batch_size, seq_len = in_indices.shape

    assert seq_len <= context_length, f"Input sequence length ({seq_len}) exceeds context_length ({context_length})"

    embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

    blocks = nn.ModuleList([
        TransformerBlock(
            embed_dim=d_model, 
            num_heads=num_heads, 
            ffn_dim=d_ff, 
            theta=rope_theta, 
            max_seq_len=context_length, 
            device=device, 
            dtype=dtype
        ) for _ in range(num_layers)
    ])

    ln_final = RMSNorm(d_model, device=device, dtype=dtype)
    lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    with torch.no_grad():
        embedding.weight.copy_(weights["token_embeddings.weight"])
        
        for i in range(num_layers):
            block = blocks[i]
            # MHA 权重 (注意这里用的是之前你类里的变量名 self.mha)
            block.mha.query.weight.copy_(weights[f"layers.{i}.attn.q_proj.weight"])
            block.mha.key.weight.copy_(weights[f"layers.{i}.attn.k_proj.weight"])
            block.mha.value.weight.copy_(weights[f"layers.{i}.attn.v_proj.weight"])
            block.mha.wo.weight.copy_(weights[f"layers.{i}.attn.output_proj.weight"])
            
            # LayerNorm 权重
            block.ln1.weight.copy_(weights[f"layers.{i}.ln1.weight"])
            block.ln2.weight.copy_(weights[f"layers.{i}.ln2.weight"])
            
            # FFN (SwiGLU) 权重
            block.ffn.w1.weight.copy_(weights[f"layers.{i}.ffn.w1.weight"])
            block.ffn.w2.weight.copy_(weights[f"layers.{i}.ffn.w2.weight"])
            block.ffn.w3.weight.copy_(weights[f"layers.{i}.ffn.w3.weight"])
            
        ln_final.weight.copy_(weights["ln_final.weight"])
        lm_head.weight.copy_(weights["lm_head.weight"])

    embedding.eval()
    blocks.eval()
    ln_final.eval()
    lm_head.eval()
    
    with torch.no_grad():
        x = embedding(in_indices)
        token_positions = torch.arange(seq_len, device=device)
        for block in blocks:
            x = block(x, token_positions=token_positions)
            
        x = ln_final(x)
        logits = lm_head(x)

    return logits

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    norm = RMSNorm(d_model, eps)
    norm.load_state_dict({"W": weights})

    return norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    ix = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([
        torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) 
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64)) 
        for i in ix
    ])

    return x, y



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    save_checkpoint(model, optimizer, iteration, out_path=out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    iteration = load_checkpoint(src_path=src, model=model, optimizer=optimizer)
    return iteration


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. 特殊 Token 处理：硬切分，确保不会跨越 Special Tokens 进行合并 [cite: 54]
    # 使用 regex.split 根据特殊 token 分割文本
    if special_tokens:
        escaped_special = [re.escape(tok) for tok in special_tokens]
        pattern = f"({'|'.join(escaped_special)})"
        # 这一步把 text 分成了 [文本, 特殊token, 文本, 特殊token...]
        chunks = [c for c in re.split(pattern, text) if c]
    else:
        chunks = [text]

    # 3. 预分词 (Pre-tokenization)
    # 使用 GPT-2 / Tiktoken 风格的正则进行初步切词 [cite: 65, 66]
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    word_counts = Counter()
    for chunk in chunks:
        # 如果是特殊 token，不要将其放进合并统计里 [cite: 71]
        if chunk in special_tokens:
            continue
        # 使用 finditer 遍历正则匹配结果，节省内存 [cite: 67]
        for match in re.finditer(PAT, chunk):
            word_counts[match.group()] += 1

    # 4. 转化为 bytes 序列
    # word_counts 形式如 {"hello": 10} -> splits 变成 { (b'h', b'e', b'l', b'l', b'o'): 10 }
    splits = {
        tuple(bytes([b]) for b in word.encode('utf-8')): count
        for word, count in word_counts.items()
    }

    # 5. 统计初始的相邻 Byte Pair 频次 [cite: 69]
    pair_counts = defaultdict(int)
    for word_tuple, count in splits.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_counts[pair] += count

    # 计算我们还需要进行多少次 merge
    # 基础词表大小为 256 + 特殊 token 的数量
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = []

    # 6. 迭代合并 (Compute BPE Merges) [cite: 68]
    for i in range(num_merges):
        if not pair_counts:
            break
            
        # 寻找频次最高的 pair
        # Tie-break 规则: 如果频次相同，取字典序 (lexicographically) 更大的 pair [cite: 73]
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        
        # 替换所有的 splits，并局部更新 pair_counts (缓存机制，极大加速) [cite: 75]
        new_splits = {}
        for word_tuple, count in splits.items():
            if best_pair[0] not in word_tuple or best_pair[1] not in word_tuple:
                new_splits[word_tuple] = count
                continue
            
            i_char = 0
            new_word = []
            while i_char < len(word_tuple):
                # 找到了可以合并的相邻对
                if (i_char < len(word_tuple) - 1 and 
                    word_tuple[i_char] == best_pair[0] and 
                    word_tuple[i_char+1] == best_pair[1]):
                    
                    merged_bytes = best_pair[0] + best_pair[1]
                    new_word.append(merged_bytes)
                    
                    # 从缓存的统计中减去旧对的频次
                    if i_char > 0:
                        pair_counts[(new_word[-2], word_tuple[i_char])] -= count
                        if pair_counts[(new_word[-2], word_tuple[i_char])] <= 0:
                            del pair_counts[(new_word[-2], word_tuple[i_char])]
                    if i_char < len(word_tuple) - 2:
                        pair_counts[(word_tuple[i_char+1], word_tuple[i_char+2])] -= count
                        if pair_counts[(word_tuple[i_char+1], word_tuple[i_char+2])] <= 0:
                            del pair_counts[(word_tuple[i_char+1], word_tuple[i_char+2])]
                            
                    # 将刚合并产生的新对的频次加上
                    if len(new_word) > 1:
                        pair_counts[(new_word[-2], new_word[-1])] += count
                        
                    i_char += 2
                else:
                    # 没碰到合并对，按原样推进
                    new_word.append(word_tuple[i_char])
                    # 检查是不是和之前合成出的节点产生了新的相邻边界
                    if len(new_word) > 1:
                        if new_word[-2] == best_pair[0] + best_pair[1] or new_word[-1] == best_pair[0] + best_pair[1]:
                             pair_counts[(new_word[-2], new_word[-1])] += count
                    i_char += 1
            
            new_splits[tuple(new_word)] = count
            
        splits = new_splits
        # best_pair 自身已经被全部合并，从统计表中清除
        if best_pair in pair_counts:
            del pair_counts[best_pair]

    # 7. 构建返回的词表 (vocab)
    vocab = {i: bytes([i]) for i in range(256)}
    current_idx = 256
    
    # 按照惯例，先加入特殊 token [cite: 80]
    for sp in special_tokens:
        vocab[current_idx] = sp.encode('utf-8')
        current_idx += 1
        
    # 再加入按创建顺序产生的 merges [cite: 71, 77]
    for m in merges:
        vocab[current_idx] = m[0] + m[1]
        current_idx += 1

    return vocab, merges
