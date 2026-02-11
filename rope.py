from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 49 (linked above).

    # Compute frequencies: theta^(-2i/d) for i in [0, head_dim/2)
    # head_dim is the full dimension, so head_dim/2 pairs
    dim = head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Compute position indices [0, 1, 2, ..., seqlen-1]
    t = torch.arange(seqlen, device=device).float()

    # Compute angles: outer product of positions and frequencies
    # freqs shape: (dim/2,), t shape: (seqlen,)
    # Result shape: (seqlen, dim/2)
    freqs = torch.outer(t, freqs)

    # Compute cos and sin
    freqs_cos = torch.cos(freqs)  # (seqlen, dim/2)
    freqs_sin = torch.sin(freqs)  # (seqlen, dim/2)

    # Reshape for broadcasting to match query/key shapes
    # query_real, query_imag shape: (batch_size, seqlen, n_heads, dim/2)
    # We need freqs_cos/sin to be broadcastable: (1, seqlen, 1, dim/2)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, dim/2)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, dim/2)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    # Apply rotation:
    # real' = real * cos - imag * sin
    # imag' = real * sin + imag * cos
    query_out_real = query_real * freqs_cos - query_imag * freqs_sin
    query_out_imag = query_real * freqs_sin + query_imag * freqs_cos

    key_out_real = key_real * freqs_cos - key_imag * freqs_sin
    key_out_imag = key_real * freqs_sin + key_imag * freqs_cos

    # Stack real and imaginary parts and reshape back to original shape
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1).flatten(-2)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1).flatten(-2)

    # Cast back to original dtype
    query_out = query_out.type_as(query)
    key_out = key_out.type_as(key)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out