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
    def __init__(self, file_path, tokenizer, max_length=1024, ratio=1):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_text = True
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['messages'])

        if 0 < ratio < 1:
            self.data = self.data[:int(len(self.data) * ratio)]

    def stat_length(self):
        all_tokens = 0
        token_lengths = []
        for data in self:
            input_ids_length = len(data['input_ids'])
            all_tokens += input_ids_length
            token_lengths.append(input_ids_length)
    
        token_lengths.sort()
        total_samples = len(self)
        mean_tokens = all_tokens / total_samples
        median_tokens = token_lengths[total_samples // 2]
        min_tokens = token_lengths[0]
        max_tokens = token_lengths[-1]
    
        print(f"Total samples: {format_number(total_samples)}")
        print(f"Total tokens: {format_number(all_tokens)}")
        print(f"Mean tokens: {format_number(mean_tokens)}")
        print(f"Median tokens: {format_number(median_tokens)}")
        print(f"Min tokens: {format_number(min_tokens)}")
        print(f"Max tokens: {format_number(max_tokens)}")

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
        # encoded = self.tokenizer.encode_plus(
        #     text,
        #     max_length=self.max_length,
        #     padding=False,
        #     truncation=True,
        #     return_tensors="pt"
        # )
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            # padding=False,
            padding="max_length",
            # padding=False,
            truncation=True,
            return_tensors="pt"
        )
        encoded["input_ids"] = encoded["input_ids"].squeeze()
        encoded["attention_mask"] = encoded["attention_mask"].squeeze()

        setattr(encoded, "text", text)
        return encoded
    
        # return_dict = {
        #     "input_ids": encoded["input_ids"].squeeze(),
        #     "attention_mask": encoded["attention_mask"].squeeze()
        # }

        # if self.include_text:
        #     return_dict["text"] = text
        
        # return return_dict