import os 
import numpy as np
import json
import random
import torch
import addition_lib
import re
import math
import argparse

import matplotlib.pyplot as plt

from addition_data_generation import generate_dataset
from optimizer import AdamW

from llama import Llama
from config import LlamaConfig

from torch.nn import functional as F
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

def create_model(model_config):
    config = LlamaConfig(
        vocab_size = model_config["vocab_size"],
        dim = model_config["dim"],
        dropout = model_config["dropout" ],
        n_layers = model_config["n_layers"],
        n_heads = model_config["n_heads"],
        n_kv_heads = model_config["n_kv_heads"],
        max_seq_len = model_config["max_seq_len"],
        layer_norm_eps = model_config["layer_norm_eps"],
        multiple_of = model_config["multiple_of"],
        hidden_dim = model_config["hidden_dim"],
        position_embedding_type = model_config["position_embedding_type"],
        use_cache = model_config["use_cache"],
    )
    model = Llama(config)
    return model 

###########################################
# ----- Model Training and Validation ----- 
###########################################
def train_one_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.

    The training loop should:
    1) Iterate over batches of tokenized input–target pairs
    2) Perform a forward pass through the model to compute logits and hidden states.
        Hidden states are not important now.
    3) Compute a cross-entropy loss between logits and target tokens
    4) Backpropagate the loss and update model parameters

    Hint:
        The model does not need to learn to reproduce the question. Since
        the input already contains the question and the task is fixed
        (addition), training capacity is better spent learning to generate
        the answer.
    Another Hint:
        token id for "=" is 12.
    """
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        # batch[0] contains input tokens, batch[1] contains target tokens
        input_ids = batch[0].to(device)
        targets = batch[1].to(device)

        # Forward pass
        logits, _ = model(input_ids, targets)

        # Compute loss only on the answer part (after "=")
        # We need to create a mask that ignores positions before and including "="
        # Create a mask for positions after "=" (token id 12)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for i in range(targets.size(0)):
            # Find position of "=" token
            equals_pos = (targets[i] == 12).nonzero(as_tuple=True)[0]
            if len(equals_pos) > 0:
                # Mask everything after "=" (we want to predict the answer)
                mask[i, equals_pos[0] + 1:] = True

        # Reshape logits and targets for loss computation
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Compute cross-entropy loss only on masked positions
        # Use ignore_index to handle any padding tokens (-1)
        if mask_flat.any():
            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], ignore_index=-1)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """
    Evaluate the model's loss on a dataset.

    The evaluation loop should:
    1) Iterate over batches of tokenized input–target pairs
    2) Mask out parts of the target sequence that correspond to the question
       (we only care about the answer for computing loss)
    3) Perform a forward pass through the model
    4) Compute cross-entropy loss, ignoring masked positions
    5) Accumulate loss over all batches and return the average

    Hint:
        The model does not need to learn to reproduce the question. Since
        the input already contains the question and the task is fixed
        (addition), training capacity is better spent learning to generate
        the answer.
    Another Hint:
        token id for "=" is 12.
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        # batch[0] contains input tokens, batch[1] contains target tokens
        input_ids = batch[0].to(device)
        targets = batch[1].to(device)

        # Forward pass
        logits, _ = model(input_ids, targets)

        # Compute loss only on the answer part (after "=")
        # Create a mask for positions after "=" (token id 12)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for i in range(targets.size(0)):
            # Find position of "=" token
            equals_pos = (targets[i] == 12).nonzero(as_tuple=True)[0]
            if len(equals_pos) > 0:
                # Mask everything after "=" (we want to predict the answer)
                mask[i, equals_pos[0] + 1:] = True

        # Reshape logits and targets for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Compute cross-entropy loss only on masked positions
        if mask_flat.any():
            loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], ignore_index=-1)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0



# ----- Saving Checkpoints and Plots -----
def create_model_filename(save_dir, epoch, dir="/checkpoints/"):
    filepath = os.path.join(save_dir, f"best_model.pth")
    return filepath 

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    dir_name = os.path.dirname(filepath)
    os.makedirs(dir_name, exist_ok=True)

    torch.save(save_info, filepath)
    print(f"Model saved to {filepath}")

def save_loss_plot(history, save_dir):

    save_path = os.path.join(save_dir , f"loss_curve.png")
    if (save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, bbox_inches="tight")

    plt.close()

def save_config(config, save_dir, filename):
    filepath = os.path.join(save_dir, f"{filename}.json")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to {filepath}")

def model_training(args):
    model_config = {
        "vocab_size": 13,
        "dim": args.dim,
        "dropout": 0.0,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "max_seq_len": 20,
        "layer_norm_eps": 1e-5,
        "multiple_of": 32,
        "hidden_dim": None,
        "position_embedding_type": "rotary",
        "use_cache": True,
    }

    training_config = {
        "capacity" : args.capacity, 
        "n_epochs" : args.epochs,
        "batch_size" : args.batch_size,
        "save_dir" : args.save_dir
    }

    # Create model 
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    model = create_model(model_config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters are {total_params}")

    # Generate datasets and loaders
    n_training_samples = math.ceil(training_config["capacity"] / (6 * total_params * training_config["n_epochs"]))
    n_validation_samples = math.ceil(n_training_samples * 0.2)
    training_config["model_parameters"] = total_params
    training_config["train_samples"] = n_training_samples

    generate_dataset(n_training_samples, "addition_train.txt", save_dir="data")
    generate_dataset(n_validation_samples, "addition_dev.txt", save_dir="data")

    train_dataset = addition_lib.create_datasets(args.train_file)
    val_dataset = addition_lib.create_datasets(args.val_file)

    train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=training_config["batch_size"], shuffle=False, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Start model training
    history = {
        "train_loss": [],
        "val_loss": []
    }

    os.makedirs(training_config["save_dir"], exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(1, training_config["n_epochs"] + 1):
        # Train the model for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Evaluate on the validation set
        val_loss = evaluate_loss(model, val_loader, device)
        
        # Append the losses to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Print the loss values
        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )

        save_loss_plot(history, training_config["save_dir"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_filepath = create_model_filename(training_config["save_dir"], epoch, dir="checkpoints")
            save_model(model, optimizer, model_config, training_config, model_filepath)
    
    save_loss_plot(history, training_config["save_dir"])
    save_config(training_config, training_config["save_dir"], "training_config")
    save_config(model_config, training_config["save_dir"], "model_config")


###########################################
# ------------- Model Testing ------------- 
###########################################
def trim_padding(tokens, pad_id=0):
    """
    Remove leading and trailing padding tokens.
    """
    tokens = tokens.tolist()
    start = 0
    end = len(tokens)
    for j in range(len(tokens)):
        if tokens[j] != pad_id:
            start = j
            break
    for j in range(len(tokens)-1, start-1, -1):
        if tokens[j] != pad_id:
            end = j+1
            break
    return tokens[start:end]

def split_before(lst, value):
    if value in lst[1:]:
        return lst[:lst.index(value)]
    return lst


def check(dataset_decode, generated_tokens):
    """
    Convert tokens to text, parse addition problem, and check correctness.
    """
    out_tokens = trim_padding(generated_tokens[0])
    out_tokens = split_before(out_tokens, 0)
    out_text = dataset_decode(out_tokens)
    # print(f"out text is {out_text}")

    try:
        m = re.match(r'(\d+)\+(\d+)=(\d+)', out_text)
        a = int(m.group(1))
        b = int(m.group(2))
        c = int(m.group(3))
        correct = (a + b) == c
    except Exception as e:
        a, b, c = -1, -1, -1
        correct = False
    return a, b, c, correct, out_text


def generate(model, prompt_tokens, max_new_tokens=10, eos_id=None, device='cpu', do_sample=False, top_k=None):
    """
    Autoregressive generation: given a prompt, generate tokens one at a time.
    """
    model.eval()
    generated = prompt_tokens.to(device).clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :]

            if do_sample:
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    sampled_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, sampled_idx)
                else:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

    return generated


def load_config(save_dir, filename):
    filepath = os.path.join(save_dir, f"{filename}")
    print(f"save dir {save_dir} filename {filename} and filepath is {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)

def load_model(save_dir, filename, model, device):
    filepath = os.path.join(save_dir, filename)
    checkpoint = torch.load(filepath, map_location=device)

    state_dict = checkpoint["model"]

    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
       if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    random.setstate(checkpoint["system_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.random.set_rng_state(checkpoint["torch_rng"].cpu())

    print(f"Loaded model from {filepath}.")
    return model

def save_experiment_info(model, model_config, train_config, accuracy, results, save_dir):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"experiment_summary.json")

    num_params = sum(p.numel() for p in model.parameters())

    predictions = [results[i][2] for i in range(len(results))]

    info = {
        "model_config": model_config,
        "train_config": train_config,
        "num_params": num_params,
        # "predictions": predictions,
        "accuracy": accuracy
    }

    with open(filepath, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Experiment info saved to {filepath}")

def save_predictions(results, save_dir):
    filepath = os.path.join(save_dir, f"predictions.json")
    predictions = [results[i][2] for i in range(len(results))]
    with open(filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {filepath}")


def test_model(
    model,
    test_dataset,
    dataset_decode,
    max_gen_len=10,
    device="cpu",
    eos_id=None,
    test_dataloader=None,
):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    incorrect = []
    results = []

    # If no dataloader provided, create one
    if test_dataloader is None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # batch[0] = input tokens
            x_prompts = batch[0]

            batch_prompt_tokens = []
            prompt_texts = []

            for x_prompt in x_prompts:
                x_prompt = trim_padding(x_prompt)
                prompt_text = test_dataset.decode(x_prompt).split("=")[0] + "="
                prompt_tokens = test_dataset.encode(prompt_text)
                prompt_tokens = torch.cat(
                    [prompt_tokens.new_tensor([0]), prompt_tokens]
                )
                batch_prompt_tokens.append(prompt_tokens)
                prompt_texts.append(prompt_text)

            batch_prompt_tokens = torch.nn.utils.rnn.pad_sequence(
                batch_prompt_tokens,
                batch_first=True,
                padding_value=0
            ).to(device)

            generated = generate(
                model,
                batch_prompt_tokens,
                max_new_tokens=max_gen_len,
                eos_id=eos_id,
                device=device
            )

            for i in range(generated.size(0)):
                a, b, c, correct_, out_text = check(
                    dataset_decode,
                    generated[i:i+1]
                )

                results.append(
                    (prompt_texts[i], generated[i], out_text, a, b, c)
                )

                if not correct_:
                    incorrect.append(
                        (prompt_texts[i], generated[i], a, b, c)
                    )

                correct += int(correct_)
                total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, incorrect, results

def model_testing(args):
    training_config = load_config(os.path.dirname(args.checkpoint), "training_config.json")
    model_config = load_config(os.path.dirname(args.checkpoint), "model_config.json")

    test_dataset = addition_lib.create_datasets(args.test_file)
    test_loader  = DataLoader(test_dataset, batch_size=training_config["batch_size"], shuffle=False)
    print(f"test length {len(test_dataset)}")

    model = create_model(model_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(
        save_dir=os.path.dirname(args.checkpoint), 
        filename=os.path.basename(args.checkpoint), 
        model=model, 
        device=device
    )
    model = model.to(device)



    accuracy, incorrect, results = test_model(model, test_dataset, test_dataset.decode, max_gen_len=10, device=device)

    print('Accuracy:', accuracy)
    print(f"test length {len(test_dataset)}")

    save_experiment_info(model, model_config, training_config, accuracy, results, os.path.dirname(args.checkpoint))
    save_predictions(results, os.path.dirname(args.checkpoint))



def get_args():
    parser = argparse.ArgumentParser(description="Addition LLaMA training and testing")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- Train --------
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--use_gpu", action="store_true")
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=1024)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--save_dir", type=str, default="addition_models/best_model/")
    train_parser.add_argument("--train_file", type=str, default="data/addition_train.txt")
    train_parser.add_argument("--val_file", type=str, default="data/addition_dev.txt")

    train_parser.add_argument("--dim", type=int, default=16)
    train_parser.add_argument("--n_layers", type=int, default=6)
    train_parser.add_argument("--n_heads", type=int, default=4)
    train_parser.add_argument("--n_kv_heads", type=int, default=4)
    train_parser.add_argument("--capacity", type=int, default=5808844800000)


    # -------- Test --------
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    test_parser.add_argument("--use_gpu", action="store_true")
    test_parser.add_argument("--seed", type=int, default=1337)
    test_parser.add_argument("--checkpoint", type=str, required=True)
    test_parser.add_argument("--test_file", type=str, default="data/addition_test.txt")
    test_parser.add_argument("--max_gen_len", type=int, default=10)
    test_parser.add_argument("--max_test_examples", type=int, default=None)

    args = parser.parse_args()
    print(f"Args: {vars(args)}")
    return args


if __name__ == "__main__":
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.command == "train":
        model_training(args)

    elif args.command == "test":
        model_testing(args)