from contextlib import nullcontext
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaPreTrainedModel, LlamaConfig
from rope import apply_rotary_emb
from utils import *


class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the LayerNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
            bias (nn.Parameter): Learnable bias parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Compute layer normalization by subtracting the mean and dividing by
        the standard deviation along the last dimension.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        # Compute mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize: (x - mean) / sqrt(var + eps)
        return (x - mean) / torch.sqrt(var + self.eps)

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying LayerNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight + self.bias

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig, causal: bool = True):
        super().__init__()
        self.causal = causal 

        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.compute_query = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.compute_key = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_value = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_output = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # causal mask (lower triangular)
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            )


    def compute_query_key_value_scores(self,
                                       query: torch.Tensor,
                                       key: torch.Tensor,
                                       value: torch.Tensor) -> torch.Tensor:
        '''
        Jointly compute Scaled Dot Product Attention (see Section 3.2.1 in
        https://arxiv.org/abs/1706.03762 for details). The query, key, and
        value tensors each have shape (bs, n_local_heads, seqlen, head_dim).

        This implementation supports *causal (autoregressive) attention*.
        When causal=True, each position i is only allowed to attend to
        positions <= i in the sequence. This is typically implemented by
        masking out future positions in the attention score matrix *before*
        applying softmax.

        An optimal implementation will compute attention for all heads
        jointly using matrix/tensor operations.
        '''
        # Compute scaled dot product: Q @ K^T / sqrt(d_k)
        # query, key, value: (bs, n_local_heads, seqlen, head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (bs, n_local_heads, seqlen, seqlen)

        # Apply causal mask if needed
        if self.causal:
            seqlen = query.size(2)
            # Mask out future positions (set them to -inf before softmax)
            scores = scores.masked_fill(self.causal_mask[:seqlen, :seqlen] == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Multiply by values
        output = torch.matmul(attn_weights, value)
        # output: (bs, n_local_heads, seqlen, head_dim)

        return output


    def forward(
        self,
        x: torch.Tensor
    ):
        '''
        Llama2 uses Grouped-Query Attention. The details of GQA are actually
        not critical to solving this assignment; you are simply asked to
        compute Scaled Dot Product Attention (see above for details). GQA is
        a memory optimization to compute multi-head attention efficiently. See
        Section 2.2 in https://arxiv.org/abs/2305.13245 or
        https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
        for details.
        '''
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        # Grouped multiquery attention: expand out keys and values.
        # Convert both to:
        # (bs, seqlen, n_local_heads, head_dim)
        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        # make heads into a batch dimension
        query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self.compute_query_key_value_scores(query, key, value)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # final projection into the residual stream
        output = self.resid_dropout(self.compute_output(output))
        return output



class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the SwiGLU activation function (see Section 2 in
        https://arxiv.org/abs/2204.02311
        '''
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = LayerNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x):
        '''
        This is the forward pass of the basic transformer building block. This is a
        modernized version of the block shown on the left of Figure 1 on
        https://arxiv.org/pdf/1706.03762.pdf.

        The transformer block should consist of:
        1) layer normalization of the input (via Root Mean Square layer normalization)
        2) self-attention on the layer-normalized input
        3) a residual connection (i.e., add the input to the output of the self-attention)
        3) layer normalization on the output of the self-attention
        4) a feed-forward network on the layer-normalized output of the self-attention
        5) add a residual connection from the unnormalized self-attention output to the
           output of the feed-forward network
        '''
        # 1-3: Layer norm -> attention -> residual connection
        h = x + self.attention(self.attention_norm(x))

        # 4-5: Layer norm -> feed-forward -> residual connection
        out = h + self.feed_forward(self.ffn_norm(h))

        return out

class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        '''
        You will probably never need to call this function, unless you decide
        to pretrain a Llama model from scratch.
        '''
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaLayer(layer_id, config))
        self.norm = LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('compute_output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return logits, h

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.95):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        We perform this generation using epsilon sampling (removing tokens below probability 
        threshold epsilon). Most likely you'll want to make sure to be in model.eval() mode 
        of operation for this. Also note this is a super inefficient version of sampling 
        with no key/value cache, but you are free to add any optimizations on top of this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            
            if temperature == 0.0:
                # select the single most likely index
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                '''
                Perform temperature sampling with top-p sampling:
                1) Scale the logits with the temperature followed by normalization using Softmax.
                2) Sort tokens by descending probability.
                3) Compute the cumulative probability distribution.
                4) Select the smallest set of tokens whose cumulative probability is >= p.
                5) Mask out all tokens outside this nucleus.
                6) Renormalize the remaining probabilities so they sum to 1.
                7) Sample from this filtered probability distribution.
                '''
                # 1. Apply temperature scaling
                logits = logits / temperature

                # 2. Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)

                # 3. Sort tokens by descending probability
                probs_sorted, probs_idx = torch.sort(probs, dim=-1, descending=True)

                # 4. Compute cumulative probability distribution
                probs_cumsum = torch.cumsum(probs_sorted, dim=-1)

                # 5. Create mask for tokens outside the nucleus
                # We want to keep tokens until cumsum exceeds top_p
                # Shift by one to keep at least one token and the first token that exceeds top_p
                mask = probs_cumsum - probs_sorted > top_p

                # 6. Mask out probabilities outside the nucleus
                probs_sorted[mask] = 0.0

                # 7. Renormalize the probabilities
                probs_sorted = probs_sorted / probs_sorted.sum(dim=-1, keepdim=True)

                # 8. Sample from the filtered distribution
                next_token_relative_idx = torch.multinomial(probs_sorted, num_samples=1)

                # 9. Map to original vocab indices
                idx_next = torch.gather(probs_idx, -1, next_token_relative_idx)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def load_pretrained(checkpoint):
  device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
  #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
  dtype = "float32"

  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
  device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

  # init from a model saved in a specific directory
  checkpoint_dict = torch.load(checkpoint, map_location=device)
  config = LlamaConfig(**checkpoint_dict['model_args'])
  model = Llama(config)
  state_dict = checkpoint_dict['model']
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict, strict=False)
  return model