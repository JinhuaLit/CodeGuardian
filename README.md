# CodeGuardian
In this project, we developed a tool that detects machine-generated code. To achieve this,
we fine-tuned the RoBerta-base model with our self-collected data and compared the results with SVM(Support Vector Machine). This repository holds content for
1. Data Collection
2. Exploratory Data Analysis
3. Model Training and Deployment 

## Demo
We developed a Flask web app to host our model. It can be accessed here: [demo](http://codecovenant.com/)
![web demo](./images/demo.png)
You can input a code snippet, and the model will classify whether it was written by a human or an AI. 
The lines that contributed the most to the prediction result will be highlighted in color.

## Requirements
Since this repo involves training and tuning the model, please install the following packages before proceeding. Alternatively, you may choose to use tools like *Google Colab*. 
```
transformers
datasets
huggingface_hub
tensorboard == 2.11
git-lfs
torchvision
tensorflow
```

## Data collection
To collect the data for human and AI written code. For the human data, we used `codeParrot/github-code` dataset 
from Huggingface (url). To generate AI-written code and ensure that the AI code and human written code are comparable, we adopted the following pipeline:
![Data Generation](./images/generation.png)

To generate AI-written code, we adopted OpenAI's api and asychronously retrived response from ChatGPT model. The prompts are stored in `llm/prompts.json` file. We then used `llm/data_generation.py` to retrieve asych responses from the api. 

## Exploratory Data Analysis
To ensure balance and fairness in our training data, we exploratorily analyzed our training data using `eda/EDA.ipynb`. Inside, we generated word cloud images and distribution of code file lengths in our data. 
You can follow the procedure in the notebook to see the outputs. 

## Training the model
### RoBerta-base
We trained our model using computing power from *Google Colab* in `roberta/roberta.ipynb`. You can follow the procedures inside to train the model. 

### SVM
The trained SVM model is included in 'SVM/SVM.ipynb.'

### Pre-trained model
To use our already fine-tuned model, you can directly use it from Huggingface. To use the pretrained tokenizer and 
model, run the following lines:

```
tokenizer = AutoTokenizer.from_pretrained("lebretou/code-human-ai")
model = AutoModelForSequenceClassification.from_pretrained("lebretou/code-human-ai")
```

To use the same encoder that we used, please download the pickle file `llm/label_encoder` with the following line:
```
# Load label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
```

