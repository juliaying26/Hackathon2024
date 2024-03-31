import os
from flask import Flask, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the pre-trained GPT-2 model and tokenizer
model_path = os.path.join('fine-tuned-model')
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_text(seed_text, max_length=100):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output_ids = input_ids.clone()

    for _ in range(max_length):
        output = model(output_ids)
        next_token_logits = output.logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sampled_token_id = torch.multinomial(next_token_probs, num_samples=1).squeeze()

        if sampled_token_id == tokenizer.eos_token_id:
            break

        output_ids = torch.cat([output_ids, sampled_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

@app.route('/generate_text', methods=['POST'])
def generate_text_api():
    seed_text = 'Princeton'
    generated_text = generate_text(seed_text)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)