# %%
import json, os, argparse
from pprint import pprint


base_dir = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description='tool for training adapters for project_cc')

parser.add_argument('-task', type=str,
                    choices=['clm','seq2seq'],
                    help='desire output format',
                    default='clm')
parser.add_argument('--out_dir', type=str, default= base_dir+"/datasets/processed/personaAI",
                    help='Path to output file ')
parser.add_argument('--in_file', type=str,
                    default=base_dir+'/datasets/raw/personachat_self_original.json',
                    help='Path to original dataset')                    

args = parser.parse_args()

# %%

with open(args.in_file,'r') as f:
    a = json.loads(f.read())

# %%
def persona_samples(file, split='train'):
    for dialogue in a[split]:
        persona = " ".join(dialogue ['personality'])
        history = dialogue['utterances'][-1]['history']
        # dio_samples = ["<persona> " + persona + "<history> " + history[i] for i in range(0,len(history),2)]
        curr_history = []
        for i in range(1,len(history),2):
            curr_history.append(history[i-1])
            yield ("<persona> " + persona + " <history> " + " ".join(curr_history), history[i])
            curr_history.append(history[i])
    

if args.task=='seq2seq':
    # %%
    with open("./datasets/processed/personaAI/train.json", 'w') as f:
        for input_text, target in persona_samples(a,split='train'):
            f.write(json.dumps({"text":input_text, "target":target} ))
            f.write("\n")
    # %%
    with open("./datasets/processed/personaAI/valid.json", 'w') as f:
        for input_text, target in persona_samples(a,split='valid'):
            f.write(json.dumps({"text":input_text, "target":target} ))
            f.write("\n")


if args.task=='clm':
    # %%
    with open("./datasets/processed/personaAI/train.txt", 'w') as f:
        for input_text, target in persona_samples(a,split='train'):
            f.write(input_text + " <answer> " + target)
            f.write("\n")
    # %%
    with open("./datasets/processed/personaAI/valid.txt", 'w') as f:
        for input_text, target in persona_samples(a,split='valid'):
            f.write(input_text + " <answer> " + target)
            f.write("\n")

# # %%
# from datasets import load_dataset
# personaAI = load_dataset('json', data_files={'train':'./datasets/processed/personaAI/train.json',
#                                         'validation':"./datasets/processed/personaAI/valid.json"}
#                         )

# # %%
# print(len(personaAI['train']))
# print(len(personaAI['validation']))


