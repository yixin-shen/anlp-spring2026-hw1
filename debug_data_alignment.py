import torch
import addition_lib

# Load dataset
dataset = addition_lib.create_datasets("data/addition_train.txt")

print("=== Checking Data Alignment ===\n")

# Check first 3 samples
for idx in range(3):
    x, y = dataset[idx]

    print(f"Sample {idx}:")
    print(f"Raw data: {dataset.words[idx]}")
    print(f"Input  x: {x.tolist()}")
    print(f"Target y: {y.tolist()}")

    # Decode
    x_decoded = dataset.decode([t for t in x.tolist() if t != 0])
    y_decoded = dataset.decode([t for t in y.tolist() if t != -1 and t != 0])

    print(f"Decoded x: {x_decoded}")
    print(f"Decoded y: {y_decoded}")

    # Check alignment
    # x should be: [0, token1, token2, ..., 0, 0, ...]
    # y should be: [token1, token2, ..., 0, -1, -1, ...]

    # Find equals sign (token 12)
    eq_pos_x = (x == 12).nonzero(as_tuple=True)[0]
    eq_pos_y = (y == 12).nonzero(as_tuple=True)[0]

    if len(eq_pos_x) > 0:
        print(f"Equals sign in x at position: {eq_pos_x[0].item()}")
    if len(eq_pos_y) > 0:
        print(f"Equals sign in y at position: {eq_pos_y[0].item()}")

    # Check what we're predicting
    print("\nWhat model should predict:")
    print("Position | Input x[i] | Target y[i] | Should predict")
    print("-" * 60)
    for i in range(min(len(x), 20)):
        if y[i] == -1:
            pred_desc = "(ignored)"
        else:
            pred_desc = f"token {y[i].item()}"
        print(f"{i:8d} | {x[i].item():10d} | {y[i].item():11d} | {pred_desc}")

    print("\n" + "="*60 + "\n")

# Check training procedure
print("\n=== Simulating Training Step ===\n")
x, y = dataset[0]
x = x.unsqueeze(0)  # Add batch dimension
y = y.unsqueeze(0)

print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Find equals position
eq_id = 12
equals_mask = (y == eq_id)
if equals_mask.any():
    eq_pos = equals_mask.float().argmax(dim=1)
    print(f"Equals position: {eq_pos.item()}")

    # Create answer mask
    seq_len = y.size(1)
    pos = torch.arange(seq_len).unsqueeze(0)
    answer_mask = pos > eq_pos.unsqueeze(1)

    print(f"Answer mask: {answer_mask[0].tolist()}")
    print(f"Positions we train on: {answer_mask[0].nonzero().squeeze().tolist()}")
    print(f"Target tokens at those positions: {y[0][answer_mask[0]].tolist()}")
