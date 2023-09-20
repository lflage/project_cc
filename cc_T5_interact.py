from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers.adapters import T5AdapterModel
from utils.utils import example_list, persona

import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
 filename='./logs/conversation_logs/'+datetime.now().strftime("%d-%m-_%H:%M:%S"))


persona = "<persona>: " + " ".join(persona)

# Task tokens
persona_tokens = ['<persona>','<history>']

# Load pre-trained BERT tokenizer from HuggingFace
tokenizer = T5TokenizerFast.from_pretrained('t5-small')
tokenizer.add_tokens(persona_tokens)

model = T5AdapterModel.from_pretrained('./persona_chat')

print('loading adapter')
model.load_adapter('./persona_chat')
print('Activating adapter')
model.set_active_adapters('persona_lm_head')


history = ' <history>: '
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

n=0
while True:
    logging.info("####### loop {} #####\n".format(n))
    example = input("")
    # print('this is example: {}'.format(example))
    # 
    history += example
    logging.info(">History: {}\n".format(history))

    text_input = persona + history
    logging.info(">text_input: {}\n".format(text_input))

    #print('this is model input: {}'.format(text_input))
    #
    model_input = tokenizer(text_input, return_tensors="pt")
    logging.info(">model_input: {}\n".format(model_input))
    #

    outputs = model.generate(input_ids=model_input.input_ids,
                            attention_mask=model_input.attention_mask,
                            max_new_tokens=40,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95
                            )
    logging.info(">out_puts: {}\n".format(outputs))

    output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #
    logging.info(">output_sentence: {}\n".format(output_sentence))
    #for i, beam_output in enumerate(outputs):
    #    print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
    #print("\n-----------------------\nThis is an output sentence:{}\n----------\n".format(output_sentence))
    print(output_sentence)
    history += output_sentence
    n+=1