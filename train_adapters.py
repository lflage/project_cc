from transformers import AdapterTrainer, TrainingArguments
from transformers import AutoModelForCausalLM, GPT2Tokenizer

import os
import json
import logging
import argparse
from datasets import Dataset


logging.basicConfig(filename=os.path.splitext(
    os.path.basename(__file__))[0]+'.log', level=logging.INFO)


parser = argparse.ArgumentParser(
    description='tool for training adapters for project_cc')

parser.add_argument('-path_to_processed', type=str,
                    default='./datasets/processed', help='path to processed dataset')
parser.add_argument('-mood', type=str, choices=['happy', 'surprised', 'sad', 'angry', 'neutral'],
                    help="""list of possible moods. 
                    ACHTUNG: If you select more than one they'll be trained sequentially""")
parser.add_argument('--out_dir', type=str, default=os.path.dirname(__file__),
                    help='Path to output file ')


args = parser.parse_args()

path_to_processed = args.path_to_processed
mood = args.mood
out_path = args.out_dir + (mood)


def jsonl_generator(shards):
    for shard in shards:
        with open(shard) as json_obj:
            for line in json_obj.readlines():
                yield json.loads(line)


jsonl_paths = []
for root, dir, files in os.walk(path_to_processed):
    for file in files:
        jsonl_paths.append(os.path.join(root, file))


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there

    model.resize_token_embeddings(len(tokenizer))
    # if num_added_tokens - orig_num_tokens > 0:
    #     print("resizing my tokens embeddings")

# Dataset processing for training and trainig functions
# Source:
# https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/06_Text_Generation.ipynb#scrollTo=ioLpFbOfnPE6
# https://adapterhub.ml/blog/2021/04/adapters-for-generative-and-seq2seq-models-in-nlp/

# Tokenize the entries in the dataset


def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    encoding = tokenizer(batch["text"], truncation=True, max_length=1024)
    # encoding = tokenizer(batch["text"])
    # For language modeling the labels need to be the input_ids
    # encoding["labels"] = encoding["input_ids"]
    return encoding


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# for line in jsonl_generator(jsonl_paths):
    # for k, v in line.items():
    #     if type(v) == float:
    #         logging.info('nan lines:\n{} : {}'.format(k, v))


moody_dataset = Dataset.from_generator(
    jsonl_generator, gen_kwargs={'shards': jsonl_paths})


cur_ds = moody_dataset.filter(
    lambda example: example["EmotionTag"] == mood).train_test_split(test_size=0.2)

del moody_dataset

print('Loading pre_trained gpt2 tokenizer')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("tokenizer len: {}".format(len(tokenizer)))
# The GPT-2 tokenizer does not have a padding token. In order to process the data
# in batches we set one here
# tokenizer.pad_token = tokenizer.eos_token

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
print("tokenizer len: {}".format(len(tokenizer)))
print('set pad token')
column_names = cur_ds["train"].column_names

dataset = cur_ds.map(encode_batch, remove_columns=column_names, batched=True)


# reate chunks with a length of block_size.
print('Creating chunks')
block_size = 50
dataset = dataset.map(group_texts, batched=True,)

dataset.set_format(type="torch", columns=[
                   "input_ids", "attention_mask", "labels"])

print(len(dataset['test'][0]['input_ids']))
###############################################################


# config = AutoConfig.from_json_file("./model/config.json")
model = AutoModelForCausalLM.from_pretrained("./model")
a = model.config
print(a)
add_special_tokens_(model, tokenizer)
b = model.config
print(b)
assert a == b
# add new adapter
model.add_adapter(mood)
# activate adapter for training
model.train_adapter(mood)


##############################################################
training_args = TrainingArguments(
    output_dir="./examples",
    do_train=True,
    remove_unused_columns=False,
    learning_rate=5e-4,
    num_train_epochs=3,
)


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)
try:
    trainer.train()
except:
    # print("didn't train")
    logging.WARNING('Did not train')

model.save_adapter("adapter_"+mood, mood)
