import re
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast

class Preprocessor:

    def __init__(self,text_file_path,batch_size,context_window):

        with open(text_file_path) as flie:
            data = flie.readlines()
        for i in range(len(data)):
            data[i] = data[i].replace("\n","")[1:-1]
        

        concatinated_data = " ".join(data)
        concatinated_data = re.sub(r'\s+', ' ', concatinated_data)
        self.all_joined_text = concatinated_data

        self.text_file_path = text_file_path
        self.batch_size = batch_size
        self.context_window = context_window

        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        

    def punctuation_processor(self):

        punctuation_pattern = r'([“”.,!?;:-])'
        self.all_joined_text = re.sub(punctuation_pattern, r' \1 ', self.all_joined_text)


    def create_train_data(self,tokenized_data):

        all_inputs = []
        all_targets = []
    
        for i in tqdm(range(len(self.all_joined_text.split())-self.context_window)):
                input_words = tokenized_data[i:i+self.context_window][:-1]
                target_word = tokenized_data[i:i+self.context_window][1:]
                
                all_inputs.append(input_words)
                all_targets.append(target_word)
        return all_inputs,all_targets
                 
    
    def tokenizer_function(self):

        trainer = BpeTrainer(vocab_size=50257, initial_alphabet=ByteLevel.alphabet(),
                                special_tokens=[
                                "<a>","<pad>","</s>","<unk>","<mask>"])

        self.tokenizer.train([self.text_file_path],trainer)
        if os.path.exists("tokenizer_gpt"):
            pass
        else:
            os.mkdir("tokenizer_gpt")
            
        self.tokenizer.save("./tokenizer_gpt/tokenizer.json")
        self.tokenizer_gpt = GPT2TokenizerFast.from_pretrained("tokenizer_gpt")
        self.tokenizer_gpt.add_special_tokens({
            "eos_token":"</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"})


        tokenized_data = self.tokenizer_gpt(self.all_joined_text)
        return tokenized_data["input_ids"]

    def fit_transform(self):

        self.punctuation_processor()
        tokenized = self.tokenizer_function()
        inputs,targets = self.create_train_data(tokenized)

        # Define generator functions
        def inputs_generator():
            for i in inputs:
                yield i

        def targets_generator():
            for i in targets:
                yield i

        dataset_inputs = tf.data.Dataset.from_generator(inputs_generator,
                output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32))
        
        dataset_targets = tf.data.Dataset.from_generator(targets_generator,
                output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32))
        dataset_inputs = dataset_inputs.batch(self.batch_size,drop_remainder=True)
        dataset_targets = dataset_targets.batch(self.batch_size,drop_remainder=True)
        
        return tf.data.Dataset.zip((dataset_inputs,dataset_targets))
