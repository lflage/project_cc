# %% 
# # Dataset Pre-Processing
# 
# The datasets are here pre-processed and all the more complex emotions are remaped
# to 5 baseline emotions:
# - happy  
# - sad  
# - angry  
# - surprised  
# - neutral  

# %%
import pandas as pd
import re, json

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# happy,sad,neutral,surprised,angry


# %%
def remove_quotation(text):
    text = text.strip()
    if text.startswith('"'):
        text = text[1:]

    if text.endswith(','):
        text = text[:-1]
    if text.endswith('"'):
        text = text[:-1]
    return text


def elf_to_df(path):
    """
    Function to read and pre-process the EmotionsLine datatset and transform
    it into a pandas dataframe
    """
    friends_dict = {}
    with open(path, 'r', encoding="utf8") as file:
        for line in file.readlines():
            line = re.sub(r'[\[\]]', "", line).strip()
            if ":" in line:
                key, value = line.split(':', 1)
                key = key.replace("\"", "").replace(',', "")
                value = value.replace(r"\u0092", "'")
                value = value.replace(r"\u0085", "...")

                value = remove_quotation(value)
                if key not in friends_dict.keys():
                    friends_dict[key] = []
                friends_dict[key].append(value)
    return pd.DataFrame(friends_dict)

def get_column_name(row):
    """
    The EmotionsGo Dataset has columns for each emotion and taggs the column for the emotion
    contained in the sentence. This function locates the tag and returns the name of the 
    column (emotion)
    """
    for column in row.index:
        if column != 'text' and row[column] == 1:
            return column
    return None

def json_lines_dump(dic,out_path):
    """
    Writes the contents of a dictionary to a jsonl file
    """
    # Convert to a list of JSON strings
    json_lines = [json.dumps(l) for l in dic]

    # Join lines and save to .jsonl file
    json_data = '\n'.join(json_lines)
    with open(out_path, 'w') as f:
        f.write(json_data)


# # Emotion lines
# %%
em_tag_map =  {
 'anger' : 'angry',
 'disgust':'angry',
 'fear':'sad',
 'joy':'happy',
 'neutral':'neutral',
 'non-neutral':'suprise',
 'sadness':'sad',
 'surprise':'surprise'}

# %%
df = elf_to_df("./datasets/raw/EmotionLines/Friends/friends_dev.json")
df1 = elf_to_df("./datasets/raw/EmotionLines/Friends/friends_test.json")
df2 = elf_to_df("./datasets/raw/EmotionLines/Friends/friends_train.json")


# %%
eml_df = pd.concat([df,df1,df2])

# %%
eml_df.drop(columns=['speaker','annotation'],inplace=True)
eml_df = eml_df.rename(mapper={'emotion':'EmotionTag','utterance':'text'},axis=1)
eml_df.dropna(inplace=True)


# %%
eml_df['EmotionTag'] = eml_df['EmotionTag'].map(em_tag_map)

# %%
em_dict = eml_df.to_dict(orient='records')

# %%
json_lines_dump(em_dict,'./datasets/processed/emotion_lines.jsonl')

# 
# # Empathetic Dialogs
# Can be loaded directly from HuggingFace

# %%
from datasets import load_dataset

emp_tag_map = {'anticipating':'happy', 'impressed':'happy',
                'guilty':'sad', 'surprised':'surprised', 
                'furious':'angry', 'excited':'happy', 'sad':'sad',
                'afraid':'sad', 'confident':'happy', 'grateful':'happy',
                'disgusted':'sad', 'jealous':'sad', 'faithful':'happy', 
                'trusting':'happy', 'prepared':'happy', 'joyful':'happy',
                'embarrassed':'sad', 'lonely':'sad', 'ashamed':'sad',
                'devastated':'sad', 'caring':'neutral', 'sentimental':'neutral', 
                'nostalgic':'neutral', 'hopeful':'happy', 'apprehensive':'sad',
                'angry':'angry', 'annoyed':'angry', 'anxious':'sad', 
                'content':'happy', 'terrified':'sad', 'proud':'happy',
                'disappointed':'neutral'}

# %%
ds = load_dataset('empathetic_dialogues')

# %%
empdio_dict = []
for j in ds:
    for i in ds[j]:
        empdio_dict.append({'text': i['utterance'].replace('_comma_',', '),
                        'EmotionTag': emp_tag_map[i["context"]]})



# %%
json_lines_dump(empdio_dict,'./datasets/processed/empathetic_dialogue.jsonl')

# %% [markdown]
# # GoEmotions

# %%
goemotion_tag_map = {'admiration':'happy',
 'amusement':'surprised',
 'anger': 'angry', 
 'annoyance':'angry',
 'approval':'happy',
 'caring':'happy',
 'confusion':'sad',
 'curiosity':'happy',
 'desire':'neutral',
 'disappointment':'sad',
 'disapproval':'angry',
 'disgust':'angry',
 'embarrassment':'sad',
 'excitement':'happy',
 'fear':'sad',
 'gratitude':'happy',
 'grief':'sad',
 'joy':'sad',
 'love':'happy',
 'nervousness':'surprised',
 'neutral':'neutral',
 'optimism':'neutral',
 'pride':'neutral',
 'realization':'neutral',
 'relief':'neutral',
 'remorse':'sad',
 'sadness':'sad',
 'surprise':'surprise'}

# %%
go1 = pd.read_csv('./datasets/raw/GoEmotions/data/full_dataset/goemotions_1.csv')
go2 = pd.read_csv('./datasets/raw/GoEmotions/data/full_dataset/goemotions_2.csv')
go3 = pd.read_csv('./datasets/raw/GoEmotions/data/full_dataset/goemotions_3.csv')

go_df = pd.concat([go1,go2,go3])

go_df = go_df.drop(columns=['id', 'author', 'subreddit', 'link_id', 'parent_id',
       'created_utc', 'rater_id', 'example_very_unclear'])


# %%
go_df['EmotionTag'] = go_df.apply(lambda row: get_column_name(row), axis=1)

# %%
go_df = go_df.drop(columns=['admiration', 'amusement', 'anger', 'annoyance', 'approval',
       'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
       'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
       'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
       'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'])

# %%
go_df.dropna(inplace=True)

# %%
go_df['EmotionTag'] = go_df['EmotionTag'].map(goemotion_tag_map)

# %%
go_dict = go_df.to_dict(orient='records')

# %%
json_lines_dump(go_dict,'./datasets/processed/go_emotions.jsonl')

# %% [markdown]
# # Concatenating

# %%
# concat_dict={}
from itertools import chain

concat_dicts = []
concat_dicts.append([go_dict,empdio_dict,em_dict])
final = chain(*concat_dicts)
# concat_dicts = go_dict.extend(empdio_dict)
# concat_dicts.extend(em_dict)
# assert len(concat_dicts) == len(empdio_dict)+len(go_dict)+len(em_dict)

# %%
final = list(chain(*final))
assert len(final) == len(go_dict) + len(empdio_dict) + len(em_dict)

# %%
# Writing the final dataset as a jsonl file
json_lines_dump(final,'./datasets/processed/emotions_ds.json')
