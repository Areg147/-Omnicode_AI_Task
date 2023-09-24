from transformers import GPT2TokenizerFast,TFGPT2LMHeadModel,GPT2Tokenizer
import argparse
import re
parser = argparse.ArgumentParser()
### example path~~~\Tuned_Inseption
parser.add_argument("--prompt", type=str, help="Promp you seed text")
### example path~~~\data\images
parser.add_argument("--model_path", type=str,  help="path to Stored Pretrained_Model")
parser.add_argument("--tokenizer_path", type=str,  help="path to Stored Tokenizer")
args = parser.parse_args()


print("-----START---LOADING---MODEL-AND-TOKENIZER")
print("\n")
Pretrained_Model = TFGPT2LMHeadModel.from_pretrained(args.model_path)
tokenizer_gpt = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
print("\n\n")
print("-----END---LOADING---MODEL-AND-TOKENIZER")
print("\n")

def printer(text):
    sentences = re.split(r'[.:]', text)
    for sentence in sentences:
        sentence = sentence.strip()  
        if sentence:  
            print(sentence)

def generate(prompt, model):
    input_token_ids = tokenizer_gpt.encode(prompt, return_tensors='tf')
    output = model.generate(input_token_ids,max_new_tokens = 45,
            do_sample =True,
            min_new_tokens = 30,
            num_beams = 5,
            no_repeat_ngram_size=2,
            num_return_sequences=1
            )
    print("\n\n\n")
    generated_text = tokenizer_gpt.decode(output[0])
    printer(generated_text)
    print("\n")

print("----GENERATION-----PROCESS-----IS-----SATRAING-----")
print("\n\n\n")

generate(args.prompt,Pretrained_Model)
