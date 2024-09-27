import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["MACA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import json
from torch.utils.data import Dataset, DistributedSampler
from transformers import AutoTokenizer
import argparse
from dataset import format_number, ChatDataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model using DDP")
    # llama3.1-8b-instruct
    parser.add_argument("--model_name", type=str, default="../Qwen2.5-14B-Instruct", help="Path to the pre-trained model")
    parser.add_argument("--train_file", type=str, default="data/iroseka_dataset.jsonl", help="Path to the training data file")
    parser.add_argument("--val_file", type=str, default="data/iroseka_validations.jsonl", help="Path to the validation data file")
    parser.add_argument("--output_dir", type=str, default="./shinku_lora", help="Output directory for saving the model")
    parser.add_argument("--max_length", type=int, default=768, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--mp_backend", type=str, default="nccl", help="Distributed backend")
    args = parser.parse_args()

    # Initialize the process group
    dist.init_process_group(backend=args.mp_backend)
    
    # Get the rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_name_short = args.model_name.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        if rank == 0:
            print(f"The tokenizer.pad_token set as a {tokenizer.eos_token}")

    dataset = {
        "train": ChatDataset(args.train_file, tokenizer, max_length=args.max_length, ratio=1),
        "validation": ChatDataset(args.val_file, tokenizer, max_length=args.max_length),
    }

    if rank == 0:
        print(dataset["train"][0].get("text", ""))
        print(dataset["train"][0])
        dataset["train"].stat_length()

    for d in dataset.values():
        d.include_text = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{rank}",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get the PEFT model
    model = get_peft_model(model, lora_config)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_{model_name_short}",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        weight_decay=0,
        dataloader_num_workers=4,
        # gradient_checkpointing=True,
        # Add DDP-specific arguments
        local_rank=int(os.environ["LOCAL_RANK"]),
        ddp_backend=args.mp_backend,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt"),
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model (only on the main process)
    if rank == 0:
        model.module.save_pretrained(f"{args.output_dir}_{model_name_short}_finetuned")
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

