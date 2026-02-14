import torch
import addition_lib
from llama import Llama
from config import LlamaConfig
import torch.nn.functional as F

# Create a tiny model
model_config = {
    "vocab_size": 13,
    "dim": 16,
    "dropout": 0.0,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 4,
    "max_seq_len": 20,
    "layer_norm_eps": 1e-5,
    "multiple_of": 32,
    "hidden_dim": None,
    "position_embedding_type": "rotary",
    "use_cache": True,
}

config = LlamaConfig(**model_config)
model = Llama(config)

# Load a small dataset
dataset = addition_lib.create_datasets("data/addition_train.txt")
print(f"Dataset size: {len(dataset)}")
print(f"Vocab size: {dataset.get_vocab_size()}")

# Get a single batch
x, y = dataset[0]
print(f"\nFirst sample:")
print(f"Input x: {x}")
print(f"Target y: {y}")
print(f"Decoded: {dataset.decode(x.tolist())}")

# Check if targets are in valid range
print(f"\nTarget range check:")
print(f"Min target: {y.min()}, Max target: {y.max()}")
print(f"Vocab size: {model_config['vocab_size']}")
print(f"Any target >= vocab_size? {(y >= model_config['vocab_size']).any()}")
print(f"Any target < 0 (besides padding)? {((y < 0) & (y != -1)).any()}")

# Forward pass
x_batch = x.unsqueeze(0)  # Add batch dimension
y_batch = y.unsqueeze(0)
print(f"\nForward pass:")
print(f"Input shape: {x_batch.shape}")
print(f"Target shape: {y_batch.shape}")

logits, _ = model(x_batch, y_batch)
print(f"Logits shape: {logits.shape}")
print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

# Compute loss on answer part
eq_id = 12
equals_mask = (y_batch == eq_id)
if equals_mask.any():
    eq_pos = equals_mask.float().argmax(dim=1)
    print(f"\nEquals position: {eq_pos.item()}")

    seq_len = y_batch.size(1)
    pos = torch.arange(seq_len).unsqueeze(0)
    answer_mask = pos > eq_pos.unsqueeze(1)

    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    y_flat = y_batch.view(-1)
    mask_flat = answer_mask.view(-1)

    # Check which positions we're training on
    valid_mask = mask_flat & (y_flat >= 0) & (y_flat < vocab_size)
    print(f"Answer positions: {answer_mask.sum().item()}")
    print(f"Valid answer positions: {valid_mask.sum().item()}")
    print(f"Target tokens in answer: {y_flat[valid_mask]}")

    # Compute loss
    if valid_mask.any():
        loss = F.cross_entropy(logits_flat[valid_mask], y_flat[valid_mask])
        print(f"Loss: {loss.item():.4f}")
else:
    print("No equals sign found!")
