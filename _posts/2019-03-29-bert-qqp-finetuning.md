---
toc: true
layout: post
description: Start of Transfer Learning Era in Natural Language Processing
categories: [BERT, NLP, deep learning]
title: BERT Fine-Tuning on Quora Question Pairs
---
# BERT Fine-Tuning on Quora Question Pairs

## Start of Transfer Learning Era in Natural Language Processing

![Photo by rawpixel on Unsplash](../images/BERT_QQP/unsplash_transfer_learning.jpeg)

BERT (Bidirectional Encoder Representations from Transformers) has started a revolution in NLP with state of the art results in various tasks, including Question Answering, GLUE Benchmark, and others. People even referred to this as [the ImageNet moment of NLP](http://ruder.io/nlp-imagenet/). If you are not familiar with BERT, please read [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/) and [BERT Paper](https://arxiv.org/abs/1810.04805).

In this blog, we will reproduce state of the art results on the Quora Question Pairs task using a pre-trained BERT model. In this task, we need to predict if the given question pair are similar or not.

## Setup

We will use Google Colab TPU runtime, which requires a GCS (Google Cloud Storage) bucket for saving models and output predictions. If you don't want to create a storage bucket, you can use GPU runtime. You can follow [this collab notebook](https://colab.research.google.com/drive/1dCbs4Th3hzJfWEe6KT-stIVDMqHZSA5V) or the copy of the notebook in [this Github repository](https://github.com/drc10723/bert_quora_question_pairs). Code snippets used in this blog might be different from the notebook for explanation purposes.

Let’s start by cloning [the BERT repository](https://github.com/google-research/bert). Alternative you can install bert-tensorflow using pip. I would recommend using the GitHub repo for better understanding. Pre-trained models are available in the GCS bucket at gs://cloud-tpu-checkpoints/bert.

```python
# cloning bert github repo
# !git clone -q https://github.com/google-research/bert.git

# add bert to sys.path  
if not 'bert' in sys.path:
  sys.path += ['bert']
  
# Instead of cloning you can install via pip also
# !pip install bert-tensorflow

# We will use base uncased model, you can give try with large models
PRETRAINED_DIR = 'gs://cloud-tpu-checkpoints/bert/'+'uncased_L-12_H-768_A-12'
```

Quora Question Pairs dataset is part of GLUE benchmark tasks. You can download the dataset from [GLUE](https://gluebenchmark.com/tasks) or [Kaggle Challenge](https://www.kaggle.com/c/quora-question-pairs/data).

## Data Loading

In Quora question pairs task, we need to predict if two given questions are similar or not. Similar pairs are labeled as 1 and non-duplicate as 0.
We will convert train, dev and test files to the list of InputExamples. Each InputExample has question1 as text_a, question2 as text_b, label, and a unique id. In the case of the test set, we will set the label to 0 for all InputExamples.

```python
# Train and Dev tsv file contains 6 tab seperated value
# We will use question1 as text_a, question2 as text_b
# and is_duplicate as label
# 'id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'
guid = "train_1"
text_a = "How can I recover old gmail account?"
text_b = "How can I access my old gmail account?"
label =  1
# creating InputExample
example = run_classifier.InputExample(guid=guid, text_a=text_a,
                                      text_b=text_b, label=label))
```

## Converting to Features

BERT uses word-piece tokenization for converting text to tokens. Tokenizer will also perform text normalization like convert all whitespace characters to spaces, lowercase the input ( uncased model) and strip out accent markers.
Let’s take an example to understand in more details.

```text
text_a: How can I recover old gmail account?
text_b: How can I access my old gmail account?

tokens: [CLS] how can i recover old gma ##il account ?
        [SEP] how can i access my old gma ##il account ? [SEP]
input_ids: 101 2129 2064 1045 8980 2214 20917 4014 4070 1029
           102 2129 2064 1045 3229 2026 2214 20917 4014 4070 1029 102
input_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  
segment_ids: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
label: 1
```

As above both questions will be tokenized and will add [CLS] as first token and [SEP] token after each question tokens. Segment ids will be 0 for question1 tokens and 1 for question2 tokens. Finally pad input_ids, input_mask, and segment_ids till max sequence length. We have used the max sequence length as 200. Zero in input_mask will represent padding.
Let’s create InputFeatures for the train set. We will save InputFeatures in the TF_Record file, which will help us in better batch loading and reduce out of memory errors.

```python
# Instantiate an instance of QQPProcessor and tokenizer
processor = QQPProcessor()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE,
                                       do_lower_case=DO_LOWER_CASE)

TRAIN_TF_RECORD = os.path.join(OUTPUT_DIR, "train.tf_record")
# getting list of train InputExample
train_examples = processor.get_train_examples(TASK_DATA_DIR)
# converting train examples to features and saving as TF Record
run_classifier.file_based_convert_examples_to_features(train_examples,
                                                       label_list,
                                                       MAX_SEQ_LENGTH,
                                                       tokenizer,
                                                       TRAIN_TF_RECORD)
```

## Creating Model

![Sentence Pair Classification tasks in BERT paper](../images/BERT_QQP/BERT_QQP.png)

Given two questions, we need to predict duplicate or not. BERT paper suggests adding extra layers with softmax as the last layer on top of the BERT model for such kinds of classification tasks. We can create an instance of the BERT model as below.

```python
# Bert model instant
model = modeling.BertModel(config=bert_config,
                           is_training=is_training,
                           input_ids=input_ids,
                           input_mask=input_mask,
                           token_type_ids=segment_ids,
                           use_one_hot_embeddings=use_one_hot_embeddings)
# Getting output for last layer of BERT
output_layer = model.get_pooled_output()
# output size for last layer
hidden_size = output_layer.shape[-1].value
```

Then we will add an NN layer with output size equal to the number of labels ( 2 in our task). For reducing overfitting, we can add the dropout layer. Finally, a softmax layer will give us probabilities for class labels. We will calculate cross entropy loss from given labels and predicted probabilities.

```python
output_weights = tf.get_variable(
    "output_weights", [num_labels, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

output_bias = tf.get_variable(
    "output_bias", [num_labels], initializer=tf.zeros_initializer())

with tf.variable_scope("loss"):
  if is_training: # 0.1 dropout
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

  # Calculate prediction probabilites
  logits = tf.matmul(output_layer, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)
  probabilities = tf.nn.softmax(logits, axis=-1)
  log_probs = tf.nn.log_softmax(logits, axis=-1)
  # Calculate loss
  one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
  per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
  loss = tf.reduce_mean(per_example_loss)
```

## Optimization and Evaluation Metrics

We will create a TPUEstimator instance for training, evaluation, and prediction, which requires model_fn. In this model_fn, we will define the optimization step for training, metrics for evaluation and loading pre-trained BERT model. BERT pre-training uses Adam with L2 regularization/ weight decay so that we will follow the same.

```python
# learning rate 2e-5
# num_train_steps = num_epoch * num_train_batches
# num_warmup_steps = 0.1 * num_train_steps
# defining optimizar function
train_op = optimization.create_optimizer(
    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

# Training estimator spec
output_spec = tf.contrib.tpu.TPUEstimatorSpec(
  mode=mode,
  loss=total_loss,
  train_op=train_op,
  scaffold_fn=scaffold_fn)
```

The initial warmup learning rate will be one-tenth of the learning rate. TPUEstimator spec will have optimization step and loss for training, metrics for evaluation and probabilities for prediction. We will calculate the following evaluation metrics:- Accuracy, Loss, F1, Precision, Recall, and AUC score.

## Creating TPUEstimator

For creating TPUEstimator, we will need model function, batch sizes ( 32, 8, 8 respectively for train, eval and predict) and config. If you are not using TPU runtime, you can set tpu_resolver to none and USE_TPU to false and TPUEstimator will fallback to GPU or CPU. Output directory should be a GCS bucket for TPU runtime. After every 1000 steps, we will save the model checkpoint.

```python
TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
# Define TPU configs
tpu_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

# Defining TPU Estimator
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=PREDICT_BATCH_SIZE)
```

## Fine-Tuning Training

For training, we need to create batches of input features. We will define an input function that will load data from the TF record file and return a batch of data generatively. We will fine-tune for three epochs. On TPU run-type, It will take about an hour.

```python
# we are using `file_based_input_fn_builder` for creating
# input function from TF_RECORD file
# same function can be used for creating input function for dev and test file
train_input_fn = run_classifier.file_based_input_fn_builder(
                                            TRAIN_TF_RECORD,
                                            seq_length=MAX_SEQ_LENGTH,
                                            is_training=True,
                                            drop_remainder=True)
# finetuning model on QQP dataset
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
```

## Evaluation

Using Estimator’s evaluate API, we can get evaluation metrics for both train and dev set.

| Metrics | Train Set | Dev Set |
| ---     | ---       | ---     |
| Loss    | 0.150     | 0.497   |
| Accuracy| 0.969     | 0.907   |
| F1      | 0.959     | 0.875   |
| AUC     | 0.969     | 0.902   |
|Precision| 0.949     | 0.864   |
|Recall   | 0.969     | 0.886   |

We are able to achieve 87.5 F1 and 90.7 % accuracy on dev set. We can further improve using the BERT large model and hyperparameter tuning. Using estimator’s predict API, we can predict for test set and custom examples. We haven’t submitted the test set for evaluation, but the BERT large model has 72.1 F1 and 89.3 % accuracy on GLUE leaderboard. Due to the different distribution of dev and test set, there is a huge difference in F1 score for both.

## Summary

Quora question pairs train set contained around 400K examples, but we can get pretty good results for the dataset (for example MRPC task in GLUE) with less than 5K examples also. BERT, OpenAI GPT, ULMFiT and many more to come will enable us to create good NLP models with few training examples.
In the end, I would recommend going through [BERT Github repository](https://github.com/google-research/bert) and medium blog [dissecting-bert](https://medium.com/dissecting-bert) for in-depth understanding.

## References

- [Google BERT Github Repo](https://github.com/google-research/bert)
- [BERT Finetuning with TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb#scrollTo=RRu1aKO1D7-Z45)
