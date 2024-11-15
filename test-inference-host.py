from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import time

app = Flask(__name__)

# Global variables to store the model and tokenizer
model = None
tokenizer = None

def load_model(base_model_path, peft_model_path):
    global model, tokenizer
    
    # Load the base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load the PEFT configuration and model
    peft_config = PeftConfig.from_pretrained(peft_model_path)
    model = PeftModel.from_pretrained(base_model, peft_model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 300)
    messages = [
        {"role": "system", "content": "Mimic 真紅's tone to continue this scenario."},
    ]

    if isinstance(prompt, list):
        for p in prompt:
            messages.append({"role": "user", "content": p})
    elif len(prompt) > 0:
        messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    text += "傍白: "
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    input_tokens = inputs.input_ids.shape[1]

    # Generate the response
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_length=max_length,
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=False,
            top_p=1,
        )
    end_time = time.time()

    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Remove the input text from the generated text
    generated_text = generated_text.replace(text, "")

    output_tokens = len(outputs[0]) - input_tokens
    inference_time = end_time - start_time
    token_speed = output_tokens / inference_time if inference_time > 0 else 0

    return jsonify({
        'input': prompt,
        'whole_input': text,
        'generated': generated_text,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'inference_time': inference_time,
        'token_speed': token_speed
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference server with a fine-tuned language model")
    parser.add_argument("--base_model", type=str, default="../Qwen2.5-14B-Instruct", help="Path to the base model")
    parser.add_argument("--peft_model", type=str, default="shinku_lora_Qwen2.5-14B-Instruct_finetuned", help="Path to the PEFT (LoRA) model")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    load_model(args.base_model, args.peft_model)
    app.run(host='0.0.0.0', port=args.port)
