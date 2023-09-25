GPT_2_from_scratch model and tokenizer https://drive.google.com/drive/folders/1vq7otsbM5jF4RrR3dZTHq8RxnEZIMV64?usp=drive_link

GPT_2_pretrained model and tokenizer  https://drive.google.com/drive/folders/1UKuPu1WKfPifV82Tuz1NAI7-DqTxgxgM?usp=drive_link


For this task, I have chosen the GPT-2 architecture because it is a unidirectional (left to right) language model, which is suitable for text generation. Since the task is to create a language model, there is no need to modify the model's architecture. Instead, we can train a tokenizer suitable for our dataset and fine-tune the pretrained model. Also, I want to mention that for fine-tuning, I added some data, and this new version of data is added to the 'finetune' folder.

In addition to fine-tuning, I have also decided to train the GPT-2 model from scratch. For measuring model performance, I have decided to use perplexity, but in most cases, language models are evaluated in downstream tasks such as GLUE tasks.

During training, the language model operates at the word level, and we generate in infference mode approximately 30-45 words, which is approximately 400-600 characters.

The model is fine-tuned using TensorFlow. Inference is performed via the terminal. First, you should install the requirements listed in requirements.txt.

Second, open the terminal and run the inference.py file. This file takes three parameters:

The first parameter is the prompt, which consists of the initial words for generating text. You can either provide the first word or words or pass an empty string.

The second parameter is the path to the pre-trained model folder, which can be downloaded from the link provided above.

The third parameter is the path to the tokenizer folder, which can also be downloaded from the link above
