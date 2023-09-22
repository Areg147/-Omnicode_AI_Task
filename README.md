𝐦𝐨𝐝𝐞𝐥 𝐚𝐧𝐝 𝐭𝐨𝐤𝐞𝐧𝐢𝐳𝐞𝐫 𝐜𝐚𝐧 𝐝𝐨𝐰𝐥𝐧𝐥𝐨𝐚𝐝 𝐟𝐫𝐨𝐦 𝐡𝐞𝐫𝐞 https://drive.google.com/drive/folders/1vq7otsbM5jF4RrR3dZTHq8RxnEZIMV64?usp=drive_link

For this task, I chose the GPT-2 architecture because it is a unidirectional (left to right) language model, which is suitable for text generation. Since the task is to create a language model, there is no need to modify the model architecture. Instead, we can train a tokenizer suitable for our dataset and fine-tune the pretrained model.

As the language model operates at the word level during inference, we generate approximately 30-45 words, which is approximately 400-600 characters. The model is fine-tuned using TensorFlow:


"Inference is performed via the terminal. First, you should install the requirements listed in requirements.txt.

Second, open the terminal and run the inference.py file. This file takes three parameters:

The first parameter is the prompt, which consists of the initial words for generating text. You can either provide the first word or words or pass an empty string.

The second parameter is the path to the pre-trained model folder, which can be downloaded from the link provided above.

The third parameter is the path to the tokenizer folder, which can also be downloaded from the link above."
