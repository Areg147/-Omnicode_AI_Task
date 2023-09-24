import pandas as pd
import re
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast

class Preprocessor:
    """
    A class for preprocessing text data from a CSV file.
        Args:
        data_path (str): Path to the CSV data file.
        approximate_N_unique_words_for_train (int): Approximate Number of words for training.(randomly sampling from uniform dist. K wrods
        index from all text corpus)(also it's Attribute)
        batch_size (int): Size of training batches.(also it's Attribute)
        max_context_window (int): Maximum context window size(also it's Attribute)

    Attributes:
        all_joined_text (str): Concatenated text from the dataset.
        tokenizer (Tokenizer): Tokenizer for text processing.
    """

    def __init__(self,data_path_csv,text_file_path,apprx_N_unique_words_for_train,batch_size,max_context_window,train_size,val_size):

        dataset = pd.read_csv(data_path_csv)
        dataset = dataset.dropna(subset = ['Player'])
        # Drop rows with missing 'Player' values
        sentences = dataset["PlayerLine"] #Concatenating all lines in the CSV file because sentences are splited

        self.all_joined_text = ' '.join(sentences)
        self.approximate_N_unique_words_for_train = apprx_N_unique_words_for_train
        self.batch_size = batch_size
        self.max_context_window = max_context_window
        self.text_file_path = text_file_path
        self.train_size =train_size
        self.val_size = val_size

         # Initialize the tokenizer with ByteLevel pre-tokenizer and decoder
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()
        

    def punctuation_processor(self):
        """
        Process punctuation in the text data by adding spaces around punctuation marks.

        This method modifies the `all_joined_text` attribute by adding spaces before and after
        common punctuation marks to ensure proper tokenization
        """
        punctuation_pattern = r'([“”.,!?;:-])'
        self.all_joined_text = re.sub(punctuation_pattern, r' \1 ', self.all_joined_text)


    def create_train_data(self,tokenized_data):
        
        """
        Create training data for the model by generating input and target pairs.

        This method generates input and target pairs from the provided tokenized data.
        For each unique starting position in the tokenized data, it creates multiple
        input sequences of varying lengths (2 to max_context_window tokens) and their corresponding
        target sequences.

        Args:.
            tokenized_data (list): The tokenized data used to create training sequences.

        Returns:
            tuple: A tuple containing two lists:
                - all_inputs (list): A list of input sequences.
                - all_targets (list): A list of corresponding target sequences.
        """
        all_inputs = []
        all_targets = []
        
        range_  = np.random.randint(0,len(tokenized_data),
                                    size=(self.approximate_N_unique_words_for_train,))
        unique = np.unique(range_)
        for i in tqdm(unique):
            for j in range(2,self.max_context_window):
                input_words = tokenized_data[i:i+j][:-1]
                target_word = tokenized_data[i:i+j][1:]
                
                all_inputs.append(input_words)
                all_targets.append(target_word)
        return all_inputs,all_targets
                 
    
    def tokenizer_function(self):
        """
        Train and configure the tokenizer for text data.

        This method trains a BytePair Encoding (BPE) tokenizer with specific settings and
        special tokens. It saves the tokenizer configuration to a file and loads it for use.
        The trained tokenizer is then used to tokenize the provided text data.

        Args:
            None

        Returns:
            list: Tokenized input IDs from the configured tokenizer.
        """

        trainer = BpeTrainer(vocab_size=50000, initial_alphabet=ByteLevel.alphabet(),
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
        """
        Fit and transform the data to create training datasets.

        This method performs several data processing steps:
        1. Processes punctuation in the text data.
        2. Tokenize the data using the 'tokenizer_function' method.
        3. Creates training data with input and target pairs.
        4. Converts the data into TensorFlow Datasets.

        Args:
            None

        Returns:
            tf.data.Dataset: A TensorFlow Dataset containing training input and target pairs.
        """
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

        dataset_inputs = dataset_inputs.padded_batch(self.batch_size,(self.max_context_window-2,),1)
        dataset_targets = dataset_targets.padded_batch(self.batch_size,(self.max_context_window-2,),1)

        final = tf.data.Dataset.zip((dataset_inputs,dataset_targets))
        train_data = final.take(self.train_size)
        validation_data = final.skip(self.train_size).take(self.val_size = val_size)
        return train_data,validation_data
