from transformers import  GPT2Config,TFGPT2LMHeadModel
from preprocessor import Preprocessor
import tensorflow as tf

class GPT(tf.keras.Model):
    def __init__(self):
        super().__init__()
        config = GPT2Config(
        vocab_size=pr.tokenizer_gpt.vocab_size,
        bos_token_id=pr.tokenizer_gpt.bos_token_id,
        eos_token_id=pr.tokenizer_gpt.eos_token_id)

        self.backbone = TFGPT2LMHeadModel(config)
    
    def call(self,input_):
        return self.backbone(input_)["logits"]
    
## Data Loading and Preparation
## defining all variables for preprocessor

csv_data_path = r"C:\Users\Dell\Desktop\Shekspir\data\Shakespeare_data.csv" #my example
txt_data_path = r"C:\Users\Dell\Desktop\Shekspir\data\alllines.txt" #my example
apprx_N_unique_words_for_train = 500 #my example
batch_size = 5 #my example
max_context_window = 6 #my example


pr = Preprocessor(csv_data_path, txt_data_path,apprx_N_unique_words_for_train,
                  batch_size, max_context_window) 
data_for_training = pr.fit_transform()

# # Creating the Model
model = GPT()
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[metric])


#Start Training
epochs = 2
model.fit(data_for_training, epochs=epochs)


