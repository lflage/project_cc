import os,argparse

import torch
from transformers import T5TokenizerFast, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.adapters import T5AdapterModel, AdapterTrainer

from itertools import chain

from datasets import load_dataset


### Logging bullshit


from transformers.utils import logging
from datetime import datetime
import wandb

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
#logging.basicConfig(level=logging.INFO,
# filename='./logs/training_logs/'+datetime.now().strftime("%d-%m-_%H:%M:%S"))


def preprocess_function(examples):
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# load dataset

personaAI = load_dataset('json', data_files={'train':'./datasets/processed/personaAI/train.json',
                                        'validation':"./datasets/processed/personaAI/valid.json"}
                        )
personaAI = personaAI.shuffle(seed=42)

parser = argparse.ArgumentParser(
    description='tool for training adapters for project_cc')

parser.add_argument('-path_to_processed', type=str,
                    default='/netscratch/fonseca/project_cc/datasets/processed/personaAI/train.txt',
                    help='path to processed dataset')
parser.add_argument('-n', type=str, help="adapter name")
parser.add_argument('-m',type=str, help="model type/name on adapters library")
parser.add_argument('-lr',type=float, help="learning rate")
parser.add_argument('-n_ep',type=int, help="nr epochs")

args = parser.parse_args()

path_to_processed = args.path_to_processed
out_path = './adapters/' + args.n



# Task tokens
persona_tokens = ['<persona_chat>','<persona>','<history>']

# Load pre-trained BERT tokenizer from HuggingFace
tokenizer = T5TokenizerFast.from_pretrained(args.m)
tokenizer.add_tokens(persona_tokens)

# Load pre-trained model from HuggingFace Hub
model = T5AdapterModel.from_pretrained(args.m)
model.resize_token_embeddings(len(tokenizer))

# adding language modeling head and setting it to train
model.add_seq2seq_lm_head(args.n,overwrite_ok=True)
model.add_adapter(args.n)
model.train_adapter(args.n)

# Pre processing and tokenizing the dataset
tokenized_personaAI = personaAI.map(preprocess_function, batched=True)

batch_size = 16
tr_args = Seq2SeqTrainingArguments(
    f"./models/{args.m}-forPersonaChat",
    learning_rate=args.lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=tr_args,
    train_dataset=tokenized_personaAI['train'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

model.save_adapter(output, args.n)
model.save_pretrained(output, "model")
