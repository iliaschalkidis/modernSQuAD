# modernSQuAD: A SQuAD-like Question Answering (QA) System

This is a single-model solution for SQuAD-like QA based on ModernBERT ([Warner et al., 2024](https://arxiv.org/abs/2412.13663)). 
ModernBERT is an up-to-date drop-in replacement for BERT-like Language Models. 
It is an Encoder-only, Pre-Norm Transformer with GeGLU activations pre-trained with Masked Language Modeling (MLM) on sequences of up to 1,024 tokens on a corpus of 2 trillion tokens of English text and code.
ModernBERT adopted many recent best practices, i.e., increased masked rating, pre-normalization, no bias terms, etc, and it also seems to have the best performance in NLU tasks among base-sized encoder-only models, like BERT, RoBERTa, DeBERTa, etc. 
The available implementation of ModernBERT also utilizes Flash Attention, which makes it substantially faster compared to the outdated implementations of the rest, e.g., ModernBERT-base seems to run 3-4x faster compared to DeBERTa-V3-base.

## Development Steps

* My implementation relies heavily on the HuggingFace API and the provided SQuAD-like QA, i.e., answer span identification, examples. 
* I implemented a SQuAD-like QA system dubbed `ModernBertForQuestionAnswering` relying on the native HuggingFace `ModernBert` implementation based on the available implementation for the HuggingFace `Roberta` model. It is available at [modeling_modernbert.py](src%2Fmodels%2Fmodernbert%2Fmodeling_modernbert.py).
* I created a simple training script [train_encoder.py](src%2Ftrainer%2Ftrain_encoder.py) relying on the HuggingFace example.
* I trained `ModernBertForQuestionAnswering` models following a brief hyperparameter grid-search for learning rate [3e-5, 5e-5]. I used a fixed batch size of 16, and a warm-up ratio of 0.05, and trained the models for 10 epochs.


## Results

| Model Name                    | Params | Eval Time (Examples / sec) | F1     | EM   |
|-------------------------------|--------|----------------------------|--------|------|
| `answerdotai/ModernBERT-base` | 149M   | 330                        | 84.5   | 81.3 |
| `microsoft/deberta-v3-base`   | 184M   | 90                         | 87.3   | 83.8 |


## Replication Steps

### Create conda environment

````shell
conda create --name squad python=3.9
pip install -r requirements
````

### Download SQuAD v2.0 locally

```shell
sh run_scripts/download_squad_v2.sh
```

### Train and Evaluate ModernBERT for SQuAD-like QA

```shell
sh run_scripts/train_encoder.sh
```


## Launch demo Streamlit app

````shell
streamlit run modernSQuAD/app.py
````

A default browser window will launch automatically and you'll be able to use the demo.