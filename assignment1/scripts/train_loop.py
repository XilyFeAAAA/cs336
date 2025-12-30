from infra import (
    Transformer, 
    AdamW, 
    cross_entropy_with_logits, 
    get_batch,
    gradient_clipping,
    lr_cosine_schedule,
    save_checkpoint,
)
from bpe import train_bpe, BPE_Tokenizer
from pathlib import Path
import numpy as np
import argparse
import torch
import random
import pickle

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_memmap(path, dtype=np.uint16):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    dataset = np.memmap(path, dtype=dtype, mode='r')
    return dataset


def train(args, train_data, val_data):
    assert args.dmodel % args.num_heads == 0
    
    device = torch.device(args.device)
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        dmodel=args.dmodel,
        d_ff=args.dff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        device=device,
        dtype=torch.float32,
        theta=args.theta,
        d_k=args.dmodel // args.num_heads,
        eps=args.eps,
        use_rope=args.use_rope
    ).to(device)
    optim = AdamW(model.parameters(), args.lr)
    
    best_losses = float("inf")
    for epoch in range(args.epochs):
        
        train_losses = []
        model.train()
        
        for iter_num in range(args.train_steps):
            
            lr = lr_cosine_schedule(
                iter_num,
                args.max_lr,
                args.min_lr,
                args.warm_up_it,
                args.cosine_it
            )
            
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            
            
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            
            optim.zero_grad()
            logits = model(x)
            loss = cross_entropy_with_logits(logits, y)
            loss.backward()
            gradient_clipping(model.parameters(), args.max_l2_norm)
            optim.step()
            train_losses.append(loss.item())

            print(f"epoch-{epoch} train loss: {train_losses[-1]:.4f}")
            
        if epoch % args.val_interval == 0:
            model.eval()
            
            val_losses = []
            with torch.no_grad():
                for _ in range(args.val_batches):
                    x, y = get_batch(val_data, args.batch_size, args.context_length, device)
                    logits = model(x)
                    loss = cross_entropy_with_logits(logits, y)
                    val_losses.append(loss.item())

            print(f"epoch-{epoch} validate loss: {val_losses[-1]:.4f}")


            if val_losses[-1] < best_losses:
                best_losses = val_losses[-1]
                filepath = Path(args.save_fp) / "checkpoint.pt"
                save_checkpoint(model, optim, iter_num, filepath)
                print("checkpoint saved")


def get_args():
    parser = argparse.ArgumentParser()
    # --- basic config ---
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--val_batches", type=int, default=100)
    # --- bpe ---
    parser.add_argument("--bpe_fp", type=str)
    parser.add_argument("--vocab_size", type=int, default=10000)
    # --- optim config ---
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warm_up_it", type=int, default=500)
    parser.add_argument("--cosine_it", type=int, default=10000)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    # --- transformer config ---
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--dmodel", type=int, default=512)
    parser.add_argument("--dff", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--theta", type=float, default=10000)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--use_rope", type=bool, default=True)
    # --- filepath ---
    parser.add_argument("--train_dataset_fp", type=str)
    parser.add_argument("--validate_dataset_fp", type=str)
    parser.add_argument("--save_fp", type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    import os

    print(os.getcwd())
    args = get_args()
    seed_everything(args.seed)
    
    
    VOCAB_FP = Path("data/vocab.pkl")
    MERGES_FP = Path("data/merges.pkl")
    special_tokens = ["<|endoftext|>"]  
    
    if VOCAB_FP.exists() and MERGES_FP.exists():
        with open(VOCAB_FP, "rb") as f:
            vocab = pickle.load(f)
        with open(MERGES_FP, "rb") as f:
            merges = pickle.load(f)
    else:
        vocab, merges = train_bpe(args.bpe_fp, args.vocab_size, special_tokens)
        print("bpe train completed")
        with open(VOCAB_FP, "wb") as f:
            pickle.dump(vocab, f)
        with open(MERGES_FP, "wb") as f:
            pickle.dump(merges, f)
    
    tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
    
    TRAIN_ENCODE_FP = Path("data/train.dat")
    VAL_ENCODE_FP = Path("data/validate.dat")
    
    if not TRAIN_ENCODE_FP.exists() or not VAL_ENCODE_FP.exists():
        with open(args.train_dataset_fp, "r", encoding="utf-8") as f:
            train_text = f.read()
        train_encode_ids = np.array(tokenizer.encode(train_text)) 
        train_encode_ids.tofile(TRAIN_ENCODE_FP)
        
        with open(args.validate_dataset_fp, "r", encoding="utf-8") as f:
            val_text = f.read()
        val_encode_ids = np.array(tokenizer.encode(val_text))
        val_encode_ids.tofile(VAL_ENCODE_FP)
    
    
    train_data = get_dataset_memmap(TRAIN_ENCODE_FP)
    val_data = get_dataset_memmap(VAL_ENCODE_FP)

    print(f"Train data size: {len(train_data)} tokens")
    print(f"Val data size: {len(val_data)} tokens")
       
    train(args, train_data, val_data)