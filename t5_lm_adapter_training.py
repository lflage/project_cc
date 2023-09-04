import os

import torch
from transformers import T5Tokenizer, TrainingArguments
from transformers.adapters import T5AdapterModel, AdapterTrainer
import adapters
from itertools import chain



out_dir = "./persona_adapter"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Load pre-trained BERT tokenizer from HuggingFace
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# An input sentence
persona = ["I'm a patient in a hospital", "I'm a 12 year old child",'I have terminal cancer']
history = ['Hi, how are you?']
answer = ["I'm fine, how about you?"]
dataset = [" ".join(["<persona>"] + persona + ["<history>"] + history + ["<answer>"]+answer)]

# Tokenize the input sentence and create a PyTorch input tensor
# input_data = tokenizer(sentence, return_tensors="pt")

# Load pre-trained BERT model from HuggingFace Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
# It can be used with different prediction heads
model = T5AdapterModel.from_pretrained('t5-small')

# adding language modeling head and setting it to train
model.add_seq2seq_lm_head('persona_persona')
model.add_adapter('persona')
model.train_adapter('persona')

max_source_length = 512
# encoding the dataset
encoding = tokenizer(
    # [task_prefix + sequence for sequence in input_sequences],
    # TODO: concatenate input strings to input form
    text=dataset,
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)
training_args = TrainingArguments(
  output_dir="./examples", 
  do_train=True,
  remove_unused_columns=False,
  learning_rate=5e-4,
  num_train_epochs=3,
)

# Trainer
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=encoding # TODO: most likely encoding will come here
)

trainer.train()
model.save_adapter("persona_chat", "persona")