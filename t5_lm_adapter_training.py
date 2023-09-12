import os

import torch
from transformers import T5TokenizerFast, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.adapters import T5AdapterModel, AdapterTrainer

from itertools import chain

from datasets import load_dataset

def preprocess_function(examples):
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

out_dir = "./persona_adapter"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# load dataset

personaAI = load_dataset('json', data_files={'train':'./datasets/processed/personaAI/train.json',
                                        'validation':"./datasets/processed/personaAI/valid.json"}
                        )


# Task tokens
persona_tokens = ['<persona>','<history>']

# Load pre-trained BERT tokenizer from HuggingFace
tokenizer = T5TokenizerFast.from_pretrained('t5-small')
tokenizer.add_tokens(persona_tokens)

# Load pre-trained model from HuggingFace Hub
model = T5AdapterModel.from_pretrained('t5-small')
model.resize_token_embeddings(len(tokenizer))

# adding language modeling head and setting it to train
model.add_seq2seq_lm_head('persona_lm_head',overwrite_ok=True)
model.add_adapter('persona_lm_head')
model.train_adapter('persona_lm_head')

# Pre processing and tokenizing the dataset
tokenized_personaAI = personaAI.map(preprocess_function, batched=True)

batch_size = 16
args = Seq2SeqTrainingArguments(
    f"T5forPersonaChat",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=100,
    predict_with_generate=True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_personaAI['train'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_adapter("persona_chat", "persona_lm_head")
model.save_pretrained("persona_chat", "model")