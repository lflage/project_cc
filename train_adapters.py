from transformers import GPT2Tokenizer
import os
import json
import logging
import argparse
from datasets import Dataset


logging.basicConfig(filename=os.path.splitext(
    os.path.basename(__file__))[0]+'.log', level=logging.INFO)


parser = argparse.ArgumentParser(
    description='tool for training adapters for project_cc')

parser.add_argument('-path_to_processed', type=int,
                    default='./datasets/processed', help='path to processed dataset')
parser.add_argument('-mood', type=str, choices=['happy', 'surprised', 'sad', 'angry', 'neutral'],
                    help="""list of possible moods. 
                    ACHTUNG: If you select more than one they'll be trained sequentially""")
parser.add_argument('--out_path', type=str, default=os.path.dirname(__file__),
                    help='Path to output file ')


args = parser.parse_args()

path_to_processed = args.path_to_processed
moods_list = args.mood
out_path = args.out_path + " ".join(moods_list)


def jsonl_generator(shards):
    for shard in shards:
        with open(shard) as json_obj:
            for line in json_obj.readlines():
                yield json.loads(line)


jsonl_paths = []
for root, dir, files in os.walk('./datasets/processed'):
    for file in files:
        jsonl_paths.append(os.path.join(root, file))

# for line in jsonl_generator(jsonl_paths):
    # for k, v in line.items():
    #     if type(v) == float:
    #         logging.info('nan lines:\n{} : {}'.format(k, v))

moody_dataset = Dataset.from_generator(
    jsonl_generator, gen_kwargs={'shards': jsonl_paths})

moody_dataset = moody_dataset.train_test_split(test_size=0.2)

# def gen(shards):
#     for shard in shards:
#         with open(shard) as f:
#             for line in f:
#                 yield {"line": line}


# shards = [f"data{i}.txt" for i in range(32)]

# ds = Dataset.from_generator(gen, gen_kwargs={"shards": jsonl_paths})

print(moody_dataset)

# def encode_batch(batch):
#   """Encodes a batch of input data using the model tokenizer."""
#   encoding = tokenizer(batch["verse_text"])
#   # For language modeling the labels need to be the input_ids
#   #encoding["labels"] = encoding["input_ids"]
#   return encoding

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# # The GPT-2 tokenizer does not have a padding token. In order to process the data
# # in batches we set one here
# tokenizer.pad_token = tokenizer.eos_token
# column_names = dataset["train"].column_names
# dataset = dataset.map(encode_batch, remove_columns=column_names, batched=True)


# from transformers import AdapterTrainingConfig, AdapterTrainer
# from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

# from adapter_transformers import AdapterType

# model_path = "path_to_your_fine_tuned_model"
# model = GPT2LMHeadModel.from_pretrained(model_path)


# adapter_names = ["task1", "task2", "task3", "task4"]

# for name in adapter_names:
#     model.add_adapter(name, AdapterType.text_task)


# task_datasets = {
#     "task1": task1_dataset,
#     "task2": task2_dataset,
#     "task3": task3_dataset,
#     "task4": task4_dataset,
# }

# training_args = AdapterTrainingConfig(
#     num_train_epochs=10,
#     learning_rate=1e-4,
# )

# for adapter_name, dataset in task_datasets.items():
#     model.train_adapter(
#         dataset=dataset, adapter_name=adapter_name, config=training_args)

# model.save_pretrained("path_to_save_model")
