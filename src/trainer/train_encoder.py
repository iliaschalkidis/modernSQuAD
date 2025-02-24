
import os
import sys

import logging
import argparse
import evaluate
from datasets import load_dataset
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions
from data import DATA_DIR, MODELS_DIR
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    default_data_collator,
    EvalPrediction,
    EarlyStoppingCallback,
    TrainingArguments
)

from src.models.modernbert.modeling_modernbert import ModernBertForQuestionAnswering

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name_or_path', default='answerdotai/ModernBERT-base', help='Model name in HF Hub')
    parser.add_argument('--do_train', default=True, type=bool, help='If true, train the model')
    parser.add_argument('--do_eval', default=True, type=bool, help='If true, evaluate the model')
    parser.add_argument('--max_length', default=512, type=int, help='The maximum total input sequence length after tokenization')
    parser.add_argument('--max_train_samples', default=None, type=int, help='If positive, limit the number of training samples')
    parser.add_argument('--max_eval_samples', default=None, type=int, help='If positive, limit the number of training samples')
    parser.add_argument('--train_batch_size', default=16, type=int, help='Batch size for training')
    parser.add_argument('--eval_batch_size', default=16, type=int, help='Batch size for evaluation')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='Learning rate')
    parser.add_argument('--max_epochs', default=10, type=int, help='Number of training epochs')
    parser.add_argument('--output_dir', default='test', type=str, help='Output directory')

    # Parse arguments
    args = parser.parse_args()

    # SetUp HuggingFace training arguments
    training_args = TrainingArguments()
    training_args.do_train = args.do_train
    training_args.do_eval = args.do_eval
    training_args.num_train_epochs = args.max_epochs
    training_args.per_device_train_batch_size = args.train_batch_size
    training_args.per_device_eval_batch_size = args.eval_batch_size
    training_args.eval_strategy = 'epoch'
    training_args.save_strategy = 'epoch'
    training_args.learning_rate = args.learning_rate
    training_args.warmup_ratio = 0.05
    training_args.output_dir = os.path.join(MODELS_DIR, args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Load HuggingFace pre-trained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, max_length=args.max_length,
                                              use_fast=True, token=True)
    if config.__class__.__name__ == 'ModernBertConfig':
        model = ModernBertForQuestionAnswering.from_pretrained(args.model_name_or_path, token=True)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path, token=True)

    # Load raw datasets
    data_files = {}
    data_files["train"] = os.path.join(DATA_DIR, 'squad_v2', 'train-v2.0.json')
    data_files["validation"] = os.path.join(DATA_DIR, 'squad_v2', 'dev-v2.0.json')
    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        field="data",
    )

    # Flatten raw datasets
    def flatten_dataset(examples):
        flattened_examples = {}
        flattened_examples['question'] = [qa['question'].lstrip() for paragraph in examples['paragraphs'] for sub_paragraph in
                                          paragraph for qa in sub_paragraph['qas']]
        flattened_examples['answer'] = [qa['answers'] for paragraph in examples['paragraphs'] for sub_paragraph in paragraph
                                        for qa in sub_paragraph['qas']]
        flattened_examples['context'] = [sub_paragraph['context'] for paragraph in examples['paragraphs'] for sub_paragraph in
                                         paragraph for _ in sub_paragraph['qas']]

        flattened_examples['id'] = [idx for idx in range(len(flattened_examples['question']))]

        return flattened_examples

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # Preprocess train dataset (tokenize + extract answer offsets)
    def preprocess_train_dataset(examples):
        # Tokenize our examples with truncation and padding, but keep the overflows
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=args.max_length,
            return_offsets_mapping=True,
            return_overflowing_tokens=False,
            padding="max_length",
        )

        # Extract spans positions (labels) from input ids
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for idx, offsets in enumerate(tokenized_examples['offset_mapping']):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][idx]
            if tokenizer.cls_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.cls_token_id)
            elif tokenizer.bos_token_id in input_ids:
                cls_index = input_ids.index(tokenizer.bos_token_id)
            else:
                cls_index = 0

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(idx)

            answer = examples["answer"][idx]
            # If no answers are given, set the cls_index as answer.
            if len(answer) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answer[0]["answer_start"]
                end_char = start_char + len(answer[0]["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case label with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if args.do_train:
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            flatten_dataset,
            batched=True,
            desc="Flatten dataset",
            remove_columns=train_dataset.column_names
        )
        if args.max_train_samples is not None:
            # Select few sample from the dataset if argument is specified
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Preprocess train dataset (tokenize + extract answer offsets)
        train_dataset = train_dataset.map(
            preprocess_train_dataset,
            batched=True,
            desc="Running tokenizer on train dataset",
            load_from_cache_file=False
        )

        # Preprocessing validation dataset (tokenize)
        def preprocess_validation_dataset(examples):
            # Tokenize our examples with truncation and padding, but keep the overflows
            tokenized_examples = tokenizer(
                examples["question" if pad_on_right else "context"],
                examples["context" if pad_on_right else "question"],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=args.max_length,
                return_offsets_mapping=True,
                return_overflowing_tokens=False,
                padding="max_length",
            )

            for idx in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example
                sequence_ids = tokenized_examples.sequence_ids(idx)
                context_index = 1 if pad_on_right else 0
                tokenized_examples["offset_mapping"][idx] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][idx])
                ]

            return tokenized_examples

    if args.do_eval:
        eval_examples = raw_datasets["validation"]
        eval_examples = eval_examples.map(
            flatten_dataset,
            batched=True,
            desc="Flatten dataset",
            remove_columns=eval_examples.column_names
        )
        if args.max_eval_samples is not None:
            # Select few sample from the dataset if argument is specified
            max_eval_samples = min(len(eval_examples), args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Preprocess validation dataset (tokenize)
        eval_dataset = eval_examples.map(
            preprocess_validation_dataset,
            batched=True,
            desc="Running tokenizer on eval dataset",
            load_from_cache_file=False
        )

    # Post-processing QA predictions
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

        references = [{"id": str(ex["id"]), "answers": ex['answer']} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Load SQuAD v2.0 metric
    metric = evaluate.load("squad_v2")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our HuggingFace Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        eval_examples=eval_examples if args.do_eval else None,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (args.max_train_samples if args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()