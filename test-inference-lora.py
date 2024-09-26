import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned language model")
    parser.add_argument("--base_model", type=str, default="../llama3.1-8b-instruct", help="Path to the base model")
    parser.add_argument("--peft_model", type=str, default="shinku_lora_llama3.1-8b-instruct/checkpoint-1989", help="Path to the PEFT (LoRA) model")
    parser.add_argument("--prompt", type=str, default="悠馬: 口の中。火傷したんだろ？", help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=300, help="Maximum length of generated text")
    args = parser.parse_args()

    # Load the base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load the PEFT configuration and model
    peft_config = PeftConfig.from_pretrained(args.peft_model)
    model = PeftModel.from_pretrained(base_model, args.peft_model)

    # Prepare the prompt
    prompt = args.prompt
    messages = [
        {"role": "system", "content": "Mimic 真紅's tone to continue this scenario."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )

    # Decode and print the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input: {prompt}")
    print(f"Whole input: {text}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()