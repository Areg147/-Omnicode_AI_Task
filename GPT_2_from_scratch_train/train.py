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


K = tf.keras.backend 

class PerplexityMetric(tf.keras.metrics.Metric):

    def __init__(self, name='perplexity', **kwargs):
        super(PerplexityMetric, self).__init__(name=name, **kwargs)
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.perplexity = self.add_weight(name='tp', initializer='zeros')

    def _calculate_perplexity(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.cross_entropy(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        step1 = K.mean(loss_, axis=-1)
        step2 = K.exp(step1)
        perplexity = K.mean(step2)

        return perplexity 


    def update_state(self, y_true, y_pred, sample_weight=None):
        
        perplexity = self._calculate_perplexity(y_true, y_pred)

        self.perplexity.assign_add(perplexity)
        
    def result(self):
        return self.perplexity

    def reset_states(self):
        self.perplexity.assign(0.)
    
## Data Loading and Preparation
## defining all variables for preprocessor

csv_data_path = "/kaggle/input/sesesese/Shakespeare_data.csv" #my example
txt_data_path = "/kaggle/input/sesesese/alllines.txt" #my example
apprx_N_unique_words_for_train = 100000 #my example
batch_size = 32 #my example
max_context_window = 35 #my example
train_size = 50000 #my example
val_size = 500 #my example


pr = Preprocessor(csv_data_path, txt_data_path,apprx_N_unique_words_for_train,
                  batch_size, max_context_window,train_size,val_size) 
data_for_training,validation_data = pr.fit_transform()

# # Creating the Model
model = GPT()
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = PerplexityMetric()

model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[metric])


#Start Training
epochs = 2
model.fit(data_for_training,validation_data=validation_data, epochs=epochs)
