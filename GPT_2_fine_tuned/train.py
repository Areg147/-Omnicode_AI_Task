from transformers import TFGPT2LMHeadModel
from preprocessor import Preprocessor
import tensorflow as tf

class GPT(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.backbone = TFGPT2LMHeadModel.from_pretrained("gpt2")
    
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
    

txt_data_path = r"C:\Users\Dell\Desktop\Shekspir\data\alllines.txt" #my example
batch_size = 16 #my example
context_window = 64 #my example


pr = Preprocessor(txt_data_path,batch_size,context_window) 
data_for_training = pr.fit_transform()

# # Creating the Model
model = GPT()
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


epochs = 2
model.fit(data_for_training, epochs=epochs)
