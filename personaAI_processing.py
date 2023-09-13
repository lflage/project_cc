# %%
import json
import pandas as pd
from pprint import pprint

# import utils.run_t5_mlm_flax

# %%

with open('./datasets/raw/personachat_self_original.json','r') as f:
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
            yield ("<persona> " + persona + "< history> " + " ".join(curr_history), history[i])
            curr_history.append(history[i])
    


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


# %%
from datasets import load_dataset
personaAI = load_dataset('json', data_files={'train':'./datasets/processed/personaAI/train.json',
                                        'validation':"./datasets/processed/personaAI/valid.json"}
                        )

# %%
print(len(personaAI['train']))
print(len(personaAI['validation']))



