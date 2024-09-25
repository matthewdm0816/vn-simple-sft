import torch
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
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def format_number(num):
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K".rstrip('0').rstrip('.')
    elif num < 1000000000:
        return f"{num/1000000:.1f}M".rstrip('0').rstrip('.')
    else:
        return f"{num/1000000000:.1f}B".rstrip('0').rstrip('.')


class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['messages'])

    def stat_length(self):
        all_tokens = 0
        for data in self:
            all_tokens += len(data['input_ids'])

        print(f"Total samples {format_number(len(self))}")
        print(f"Total tokens {format_number(all_tokens)}")

    def __len__(self):
        return len(self.data)
    
    def _get_text(self, messages):
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text

    def __getitem__(self, idx):
        messages = self.data[idx]
        
        # Apply chat template
        text = self._get_text(messages)
        
        # Tokenize and truncate
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "text": text,
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze()
        }


model_name = "../Qwen2.5-1.5B-Instruct"
model_name_short = model_name.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = {
    "train": ChatDataset("data/iroseka_dataset.jsonl", tokenizer, max_length=1200),
    "validation": ChatDataset("data/iroseka_validations.jsonl", tokenizer, max_length=1200),
}
print(dataset["train"][0].get("text", ""))
dataset["train"].stat_length()
# exit(0)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# Prepare the model for k-bit training
# model = prepare_model_for_kbit_training(model)

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


# Set up the training arguments
num_train_epochs = 3
training_args = TrainingArguments(
    output_dir=f"./shinku_{model_name_short}_lora",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=3e-5,
    # fp16=True,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    eval_strategy="steps",
    eval_steps=0.5//num_train_epochs,
    weight_decay=0,
    dataloader_num_workers=4,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(f"./shinku_{model_name_short}_lora_finetuned")
