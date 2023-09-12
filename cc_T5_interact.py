from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers.adapters import T5AdapterModel
from utils.utils import example_list, persona

persona = "<persona>: " + " ".join(persona)

# Task tokens
persona_tokens = ['<persona>','<history>']

# Load pre-trained BERT tokenizer from HuggingFace
tokenizer = T5TokenizerFast.from_pretrained('t5-small')
tokenizer.add_tokens(persona_tokens)

model = T5AdapterModel.from_pretrained('./persona_chat')

model.load_adapter('./persona_chat')
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

while True:
    example = input(">>>> ")
    # print('this is example: {}'.format(example))
    # 
    history += example
    text_input = persona + history
    #print('this is model input: {}'.format(text_input))
    #
    model_input = tokenizer(text_input, return_tensors="pt")
    #
    outputs = model.generate(input_ids=model_input.input_ids)
    output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #

    #print("\n-----------------------\nThis is an output sentence:{}\n----------\n".format(output_sentence))
    print(output_sentence)
    history += output_sentence