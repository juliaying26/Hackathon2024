# -*- coding: utf-8 -*-
"""EisgruberDolanLSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p9S3BF6LzKQHtW57RbeCNyxw_36htBHM
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Load the pre-trained GPT-2 tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)  # Load the pre-trained GPT-2 model and move it to the device

# Create an optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)  # Create an AdamW optimizer with a learning rate of 5e-5 for the model's parameters

# Function to preprocess the input text
def preprocess_text(text):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)  # Tokenize the input text, convert it to PyTorch tensor, and move it to the device
    return input_ids  # Return the tokenized input as PyTorch tensor

# Function to generate new sentences
def generate_text(model, tokenizer, seed_text, max_length=100):
    input_ids = preprocess_text(seed_text)  # Preprocess the seed text
    output_ids = input_ids.clone()  # Create a copy of the input_ids tensor for output

    for _ in range(max_length):
        output = model(output_ids)  # Pass the output_ids tensor to the model to get the output
        next_token_logits = output.logits[:, -1, :]  # Get the logits for the next token from the model output
        next_token_probs = torch.softmax(next_token_logits, dim=-1)  # Convert the logits to probabilities using softmax
        sampled_token_id = torch.multinomial(next_token_probs, num_samples=1).squeeze()  # Sample the next token ID from the probability distribution

        if sampled_token_id == tokenizer.eos_token_id:  # Check if the sampled token is the end-of-sequence token
            break  # If it is, break out of the loop

        output_ids = torch.cat([output_ids, sampled_token_id.unsqueeze(0).unsqueeze(0)], dim=1)  # Concatenate the sampled token ID to the output_ids tensor

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # Decode the generated output_ids tensor to get the generated text
    return generated_text  # Return the generated text

df = pd.read_csv('DeanDolan.csv',header = None)

input_sentences = df[0].tolist()
# copy = df[0].tolist()
# input_sentences = input_sentences + copy

# Fine-tune the model with the input sentences
model.train()  # Set the model to training mode
for sentence in input_sentences:
    input_ids = preprocess_text(sentence)  # Preprocess the input sentence
    outputs = model(input_ids, labels=input_ids)  # Pass the input_ids to the model and get the output with labels (for training)
    loss = outputs.loss  # Get the loss from the output
    loss.backward()  # Compute the gradients of the loss
    optimizer.step()  # Update the model's parameters using the optimizer
    optimizer.zero_grad()  # Reset the gradients to zero

# Save the fine-tuned model
model.save_pretrained('path/to/save/fine-tuned-model')

model2 = GPT2LMHeadModel.from_pretrained('path/to/save/fine-tuned-model').to(device)

# Generate new sentences based on the input data
seed_text = "Princeton"  # Seed text for generating new sentences
generated_text = generate_text(model, tokenizer, seed_text)  # Generate new text using the fine-tuned model
print(generated_text)  # Print the generated text

!pip install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
!pip install torch
!pip install spacy
!python -m spacy download en

from gramformer import Gramformer
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)
gf = Gramformer(models = 1, use_gpu=False)
mytext = """I is testng grammar tool using python. It does not costt anythng."""
gf.correct(mytext, max_candidates=1)

mytext = "Princeton CC students are literally demonstrating their pride.“ You manage to make history.’ You are no better than others., and those whose lives, works, and lives are so meaningful.”You spent your days on the internet quite literally fading.”.” You, like the people who you love to see—Thank you for offering these moments. Do of course, this connection have you ever attempted and resist?"

gf.correct(generated_text.replace("“","").replace(",","").replace("‘",""), max_candidates=1)

!pip freeze