
from transformers.adapters import T5AdapterModel
from transformers import T5TokenizerFast
from utils.utils import example_list, persona

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
 filename='./logs/conversation_logs/'+datetime.now().strftime("%d-%m-_%H:%M:%S"))


persona = "<persona>: " + " ".join(persona)

# Task tokens
persona_tokens = ['<persona_chat>','<persona>','<history>']

# Load pre-trained T5 tokenizer from HuggingFace
tokenizer = T5TokenizerFast.from_pretrained('t5-small')
# Resizing tokenizer
tokenizer.add_tokens(persona_tokens)

# Load pre-trained model
model = T5AdapterModel.from_pretrained('t5-small')
# resizing token embeddings
model.resize_token_embeddings(len(tokenizer))

print('loading adapter')
model.load_adapter('./adapters/persona_chat_final')
print('Activating adapter')
model.set_active_adapters('persona_chat_final')


history = ' <history>: '
# Commented out bellow is the test loop with exeample sentences
#for example in example_list:
#    print('this is example: {}'.format(example))
#    # 
#    history += example
#    text_input = persona + history
#    print('this is model input: {}'.format(text_input))
#    #
#    model_input = tokenizer(text_input, return_tensors="pt")
#    #
#    outputs = model.generate(input_ids=model_input.input_ids)
#    output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
#    #
#
#    print("\n-----------------------\nThis is an output sentence:{}\n----------\n".format(output_sentence))
#    history += output_sentence


# conversation Loop
n=0
print('start talking')
while True:
    logging.info("####### loop {} #####\n".format(n))
    example = input(">>>")
    
    example = example.split(">>>")[0]
  
    history += example + " "
    logging.info(">History: {}\n".format(history))

    text_input = persona + history
    logging.info(">text_input: {}\n".format(text_input))

    model_input = tokenizer(text_input, return_tensors="pt")
    logging.info(">model_input: {}\n".format(model_input))

    # outputs = model.generate(input_ids=model_input.input_ids,
    #                         attention_mask=model_input.attention_mask,
    #                         max_new_tokens=40,
    #                         do_sample=True,
    #                         no_repeat_ngram_size=2,
    #                         top_k=3,
    #                         temperature=0.7
    #                         )
    outputs = model.generate(input_ids=model_input.input_ids,
                        attention_mask=model_input.attention_mask,
                        max_new_tokens=40,
                        do_sample=True,
                        top_k=15,
                        temperature=0.96
                        )
    

    logging.info(">outputs: {}\n".format(outputs))

    output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(">output_sentence: {}\n".format(output_sentence))
    
    print(output_sentence)

    history += output_sentence + " "
    n+=1
