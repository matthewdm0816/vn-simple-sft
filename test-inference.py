import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def generate_text(prompt, model, tokenizer, max_length=512):
    messages = [
        {"role": "system", "content": "You are 'Apeiria', a world-class AI system. You are a helpful assistant. You reply in Chinese."},
        {"role": "user", "content": prompt},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    input_tokens = count_tokens(text, tokenizer)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    start_time = time.time()
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length,
    )
    
    end_time = time.time()
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    output_tokens = count_tokens(response, tokenizer)
    generation_time = end_time - start_time
    token_speed = output_tokens / generation_time if generation_time > 0 else 0
    
    return response, input_tokens, output_tokens, generation_time, token_speed

def main():
    # Specify the model name
    model_name = "../Qwen2.5-14B-Instruct"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )

    # Set the model to evaluation mode
    model.eval()

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to cook.",
        "Summarize the main causes of climate change.",
        "How does photosynthesis work?",
        "Describe the process of making traditional Italian pizza.",
        "What are the key differences between Python and Java programming languages?",
        "Explain the theory of relativity in layman's terms.",
        "Write a haiku about artificial intelligence.",
        "What were the main causes of World War II?",
        # ... (add more prompts as desired)
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    total_generation_time = 0

    # Generate and print responses for each prompt
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response, input_tokens, output_tokens, generation_time, token_speed = generate_text(prompt, model, tokenizer)
        print(f"Response: {response}")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Token generation speed: {token_speed:.2f} tokens/second\n")

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_generation_time += generation_time

    # Print overall statistics
    print("Overall Statistics:")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total generation time: {total_generation_time:.2f} seconds")
    print(f"Average token generation speed: {total_output_tokens / total_generation_time:.2f} tokens/second")
    print(f"Average input tokens per prompt: {total_input_tokens / len(prompts):.2f}")
    print(f"Average output tokens per prompt: {total_output_tokens / len(prompts):.2f}")
    print(f"Average generation time per prompt: {total_generation_time / len(prompts):.2f} seconds")

if __name__ == "__main__":
    main()
