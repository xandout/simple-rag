"""
LoRA training script for instruction-tuning chat models from a JSON dataset.

Defaults to google/gemma-3-1b-it and expects a JSON file with a list of
objects containing at least {"prompt": str, "response": str} fields.

Example usage:
    python runpod/new/train_lora.py \
        --data ./runpod/new/pokemon.json \
        --output_dir ./adapters/gemma3-1b-it-lora

To run with different hyperparameters:
    python runpod/new/train_lora.py \
        --data ./data/train.json \
        --model google/gemma-3-1b-it \
        --r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --batch_size 4 --grad_accum 4 --lr 2e-4 --epochs 3 --max_len 2048

The resulting LoRA adapter can be loaded at inference with vLLM using:
    vllm serve google/gemma-3-1b-it --lora-modules lora_adapter=<OUTPUT_DIR>
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def read_json_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects with 'prompt' and 'response'.")
    cleaned = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        # Support multiple common field names
        prompt = (
            item.get("prompt")
            or item.get("instruction")
            or item.get("input")
            or item.get("question")
        )
        response = item.get("response") or item.get("output") or item.get("answer")
        if not prompt or not response:
            continue
        cleaned.append({"prompt": str(prompt), "response": str(response)})
    if not cleaned:
        raise ValueError("No valid items found. Expected fields like 'prompt' and 'response'.")
    return cleaned


def build_chat_texts(examples: List[Dict], tokenizer: AutoTokenizer) -> List[str]:
    texts: List[str] = []
    add_system = False  # Keep it minimal; rely on model's IT priors
    system_msg = "You are a helpful assistant."
    for ex in examples:
        messages = []
        if add_system:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": ex["prompt"]})
        messages.append({"role": "assistant", "content": ex["response"]})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts


@dataclass
class TextDataset:
    texts: List[str]
    tokenizer: AutoTokenizer
    max_len: int

    def to_hf(self) -> Dataset:
        ds = Dataset.from_dict({"text": self.texts})

        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                max_length=self.max_len,
                truncation=True,
                padding=False,
                return_attention_mask=True,
            )

        return ds.map(tokenize, batched=True, remove_columns=["text"]).shuffle(seed=42)


def train_lora(
    data_path: str,
    model_name: str,
    output_dir: str,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    epochs: int,
    max_len: int,
    fp16: bool,
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading dataset from:", data_path)
    raw = read_json_dataset(data_path)
    texts = build_chat_texts(raw, tokenizer)

    # Split train/val
    split_idx = max(1, int(0.95 * len(texts)))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    if not val_texts:
        val_texts = texts[-1:]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else None,
        device_map="auto",
    )

    # Optional: prepare 4/8-bit if user provides such a model; otherwise no-op
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=None,  # let PEFT auto-select common attention proj modules
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = TextDataset(train_texts, tokenizer, max_len=max_len).to_hf()
    val_ds = TextDataset(val_texts, tokenizer, max_len=max_len).to_hf()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=min(batch_size, 4),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        bf16=not fp16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save only the adapter weights in huggingface PEFT format
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters from a JSON dataset.")
    parser.add_argument("--data", required=True, help="Path to JSON file with prompt/response pairs.")
    parser.add_argument("--model", default="google/gemma-3-1b-it", help="Base model name or path.")
    parser.add_argument("--output_dir", default="./adapters/gemma3-1b-it-lora", help="Where to save the LoRA adapter.")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_len", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_lora(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output_dir,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        max_len=args.max_len,
        fp16=bool(args.fp16),
    )


if __name__ == "__main__":
    main()