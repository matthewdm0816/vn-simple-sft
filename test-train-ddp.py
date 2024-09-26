import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

from dataset import format_number, ChatDataset
    

def main():
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Get the rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_name = "../Qwen2.5-7B-Instruct"
    model_name_short = model_name.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = {
        "train": ChatDataset("data/iroseka_dataset.jsonl", tokenizer, max_length=768, ratio=1),
        "validation": ChatDataset("data/iroseka_validations.jsonl", tokenizer, max_length=768),
    }

    if rank == 0:
        print(dataset["train"][0].get("text", ""))
        print(dataset["train"][0])
        dataset["train"].stat_length()

    for d in dataset.values():
        d.include_text = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{rank}",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get the PEFT model
    model = get_peft_model(model, lora_config)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])

    # Set up the training arguments
    num_train_epochs = 3
    per_device_train_batch_size = 1
    per_device_eval_batch_size = per_device_train_batch_size
    training_args = TrainingArguments(
        output_dir=f"./shinku_{model_name_short}_lora",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=10,
        # save_strategy="steps",
        # eval_strategy="steps",
        # eval_steps=0.5 // num_train_epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        weight_decay=0,
        dataloader_num_workers=0,
        # Add DDP-specific arguments
        local_rank=int(os.environ["LOCAL_RANK"]),
        ddp_backend="nccl",
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
        model.module.save_pretrained(f"./shinku_{model_name_short}_lora_finetuned")
        dist.destroy_process_group()

if __name__ == "__main__":
    main()