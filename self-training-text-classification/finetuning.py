# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning the library models for sequence classification."""

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import shutil
from typing import List, Optional

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import ExplicitEnum
from transformers.trainer_utils import IntervalStrategy

# Import accelerator
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

logger = logging.getLogger(__name__)


class Split(ExplicitEnum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"
    INFER = "infer"


@dataclasses.dataclass
class FTModelArguments:
    """Arguments pertaining to which config/tokenizer/model we are going to fine-tune from."""

    model_name_or_path: str = dataclasses.field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    use_fast_tokenizer: Optional[bool] = dataclasses.field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    cache_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co."},
    )


@dataclasses.dataclass
class FTDataArguments:
    """Arguments pertaining to what data we are going to input our model for training and evaluation."""

    train_file: str = dataclasses.field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    eval_file: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    infer_file: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "A csv or a json file containing the data to predict on."}
    )
    task_name: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "The name of the task to train on."},
    )
    label_list: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"help": "The list of labels for the task."}
    )

    max_length: Optional[int] = dataclasses.field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: Optional[bool] = dataclasses.field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )


@dataclasses.dataclass
class FTTrainingArguments:
    """Training arguments pertaining to the training loop itself."""

    output_dir: str = dataclasses.field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    do_train: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "Whether to run training or not."},
    )
    do_eval: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "Whether to run evaluation on the validation set or not."},
    )
    do_predict: Optional[bool] = dataclasses.field(
        default=False,
        metadata={"help": "Whether to run inference on the inference set or not."},
    )
    seed: Optional[int] = dataclasses.field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    num_train_epochs: Optional[int] = dataclasses.field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."},
    )
    per_device_train_batch_size: Optional[int] = dataclasses.field(
        default=8,
        metadata={"help": "The batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: Optional[int] = dataclasses.field(
        default=8,
        metadata={"help": "The batch size per GPU/TPU core/CPU for evaluation."},
    )
    weight_decay: Optional[float] = dataclasses.field(
        default=0.0,
        metadata={
            "help": (
                "The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in"
                " [`AdamW`] optimizer."
            )
        },
    )
    learning_rate: Optional[float] = dataclasses.field(
        default=5e-5,
        metadata={"help": "The initial learning rate for [`AdamW`] optimizer."},
    )
    gradient_accumulation_steps: Optional[int] = dataclasses.field(
        default=1,
        metadata={
            "help": (
                "Number of updates steps to accumulate the gradients for, before performing a backward/update pass."
            )
        },
    )
    max_steps: Optional[int] = dataclasses.field(
        default=-1,
        metadata={
            "help": (
                "If set to a positive number, the total number of training steps to perform. Overrides"
                " `num_train_epochs`."
            )
        },
    )
    lr_scheduler_type: Optional[str] = dataclasses.field(
        default="linear", metadata={"help": "The scheduler type to use."}
    )
    warmup_steps: Optional[int] = dataclasses.field(
        default=0,
        metadata={
            "help": (
                "Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of"
                " `warmup_ratio`."
            )
        },
    )
    eval_strategy: Optional[str] = dataclasses.field(
        default="no",
        metadata={
            "help": 'The evaluation strategy to adopt during training. Possible values are: ["no", "step", "epoch]'
        },
    )
    eval_steps: Optional[int] = dataclasses.field(
        default=500,
        metadata={"help": 'Number of update steps between two evaluations if `eval_strategy="steps"`.'},
    )
    eval_metric: Optional[str] = dataclasses.field(
        default="accuracy", metadata={"help": "The evaluation metric used for the task."}
    )
    keep_checkpoint_max: Optional[int] = dataclasses.field(
        default=1,
        metadata={"help": "The maximum number of best checkpoint files to keep."},
    )
    early_stopping_patience: Optional[int] = dataclasses.field(
        default=10,
        metadata={"help": "Number of evaluation calls with no improvement after which training will be stopped."},
    )
    early_stopping_threshold: Optional[float] = dataclasses.field(
        default=0.0,
        metadata={
            "help": "How much the specified evaluation metric must improve to satisfy early stopping conditions."
        },
    )


def train(args, accelerator, model, tokenizer, train_dataloader, optimizer, lr_scheduler, eval_dataloader=None):
    """Train a model on the given training data."""

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", args.num_examples[Split.TRAIN.value])
    logger.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_local_main_process)

    checkpoints = None
    eval_results = None
    best_checkpoint = None
    best_eval_result = None
    early_stopping_patience_counter = 0
    should_training_stop = False
    epoch = 0
    completed_steps = 0
    train_loss = 0.0
    model.zero_grad()

    for _ in range(args.num_train_epochs):
        epoch += 1
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                # Evaluate during training
                if (
                    eval_dataloader is not None
                    and args.eval_strategy == IntervalStrategy.STEPS.value
                    and args.eval_steps > 0
                    and completed_steps % args.eval_steps == 0
                ):
                    accelerator.wait_for_everyone()
                    new_checkpoint = f"checkpoint-{IntervalStrategy.STEPS.value}-{completed_steps}"
                    new_eval_result = evaluate(args, accelerator, eval_dataloader, "eval", model, new_checkpoint)[
                        args.eval_metric
                    ]
                    logger.info(
                        "Evaluation result at step %d: %s = %f", completed_steps, args.eval_metric, new_eval_result
                    )
                    
                    # Initialize or update checkpoint tracking
                    if checkpoints is None:
                        checkpoints = np.array([new_checkpoint])
                        eval_results = np.array([new_eval_result])
                        best_checkpoint = new_checkpoint
                        best_eval_result = new_eval_result
                    else:
                        # Check if this is a new best result
                        if best_eval_result is None or (new_eval_result - best_eval_result) > args.early_stopping_threshold:
                            best_checkpoint = new_checkpoint
                            best_eval_result = new_eval_result
                            early_stopping_patience_counter = 0
                        else:
                            if abs(new_eval_result - best_eval_result) < 1e-9:  # Consider equal
                                best_checkpoint = new_checkpoint
                                best_eval_result = new_eval_result
                            early_stopping_patience_counter += 1

                        # Check early stopping condition
                        if early_stopping_patience_counter >= args.early_stopping_patience:
                            should_training_stop = True

                        # Update tracking arrays
                        checkpoints = np.append(checkpoints, [new_checkpoint], axis=0)
                        eval_results = np.append(eval_results, [new_eval_result], axis=0)
                        
                        # Sort by evaluation results (best first)
                        sorted_ids = np.argsort(-eval_results)  # Negative for descending order
                        eval_results = eval_results[sorted_ids]
                        checkpoints = checkpoints[sorted_ids]

                    # Remove excess checkpoints
                    if len(checkpoints) > args.keep_checkpoint_max:
                        # Delete the worst checkpoints
                        for checkpoint_to_remove in checkpoints[args.keep_checkpoint_max:]:
                            if checkpoint_to_remove != new_checkpoint:
                                if accelerator.is_main_process:
                                    shutil.rmtree(os.path.join(args.output_dir, checkpoint_to_remove), ignore_errors=True)
                        # Keep only the top checkpoints
                        checkpoints = checkpoints[:args.keep_checkpoint_max]
                        eval_results = eval_results[:args.keep_checkpoint_max]

                    # Save current checkpoint if it's in the top ones
                    if new_checkpoint in checkpoints:
                        checkpoint_output_dir = os.path.join(args.output_dir, new_checkpoint)
                        if accelerator.is_main_process:
                            if not os.path.exists(checkpoint_output_dir):
                                os.makedirs(checkpoint_output_dir)
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=accelerator.save)
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(checkpoint_output_dir)
                            logger.info("Saving model checkpoint to %s", checkpoint_output_dir)

            if completed_steps >= args.max_steps:
                break

            if should_training_stop:
                break

        # Evaluate during training at epoch end
        if eval_dataloader is not None and args.eval_strategy == IntervalStrategy.EPOCH.value and not should_training_stop:
            accelerator.wait_for_everyone()
            new_checkpoint = f"checkpoint-{IntervalStrategy.EPOCH.value}-{epoch}"
            new_eval_result = evaluate(args, accelerator, eval_dataloader, "eval", model, new_checkpoint)[
                args.eval_metric
            ]
            logger.info("Evaluation result at epoch %d: %s = %f", epoch, args.eval_metric, new_eval_result)

            # Initialize or update checkpoint tracking
            if checkpoints is None:
                checkpoints = np.array([new_checkpoint])
                eval_results = np.array([new_eval_result])
                best_checkpoint = new_checkpoint
                best_eval_result = new_eval_result
            else:
                # Check if this is a new best result
                if best_eval_result is None or (new_eval_result - best_eval_result) > args.early_stopping_threshold:
                    best_checkpoint = new_checkpoint
                    best_eval_result = new_eval_result
                    early_stopping_patience_counter = 0
                else:
                    if abs(new_eval_result - best_eval_result) < 1e-9:  # Consider equal
                        best_checkpoint = new_checkpoint
                        best_eval_result = new_eval_result
                    early_stopping_patience_counter += 1

                # Check early stopping condition
                if early_stopping_patience_counter >= args.early_stopping_patience:
                    should_training_stop = True

                # Update tracking arrays
                checkpoints = np.append(checkpoints, [new_checkpoint], axis=0)
                eval_results = np.append(eval_results, [new_eval_result], axis=0)
                
                # Sort by evaluation results (best first)
                sorted_ids = np.argsort(-eval_results)  # Negative for descending order
                eval_results = eval_results[sorted_ids]
                checkpoints = checkpoints[sorted_ids]

            # Remove excess checkpoints
            if len(checkpoints) > args.keep_checkpoint_max:
                # Delete the worst checkpoints
                for checkpoint_to_remove in checkpoints[args.keep_checkpoint_max:]:
                    if checkpoint_to_remove != new_checkpoint:
                        if accelerator.is_main_process:
                            shutil.rmtree(os.path.join(args.output_dir, checkpoint_to_remove), ignore_errors=True)
                # Keep only the top checkpoints
                checkpoints = checkpoints[:args.keep_checkpoint_max]
                eval_results = eval_results[:args.keep_checkpoint_max]

            # Save current checkpoint if it's in the top ones
            if new_checkpoint in checkpoints:
                checkpoint_output_dir = os.path.join(args.output_dir, new_checkpoint)
                if accelerator.is_main_process:
                    if not os.path.exists(checkpoint_output_dir):
                        os.makedirs(checkpoint_output_dir)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(checkpoint_output_dir)
                    logger.info("Saving model checkpoint to %s", checkpoint_output_dir)

        if completed_steps >= args.max_steps:
            break

        if should_training_stop:
            break

    # Handle best checkpoint
    if best_checkpoint is not None:
        logger.info("Best checkpoint: %s", best_checkpoint)
        logger.info("Best evaluation result: %s = %f", args.eval_metric, best_eval_result)
        best_checkpoint_output_dir = os.path.join(args.output_dir, best_checkpoint)
        best_final_output_dir = os.path.join(args.output_dir, "best-checkpoint")
        
        if accelerator.is_main_process:
            if os.path.exists(best_checkpoint_output_dir):
                if os.path.exists(best_final_output_dir):
                    shutil.rmtree(best_final_output_dir, ignore_errors=True)
                shutil.move(best_checkpoint_output_dir, best_final_output_dir)
            # Clean up other checkpoints
            for item in os.listdir(args.output_dir):
                item_path = os.path.join(args.output_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-") and item != "best-checkpoint":
                    shutil.rmtree(item_path, ignore_errors=True)
        accelerator.wait_for_everyone()

    else:
        # Save final model if no checkpoints were saved during training
        checkpoint_output_dir = os.path.join(args.output_dir, "best-checkpoint")
        if accelerator.is_main_process:
            if not os.path.exists(checkpoint_output_dir):
                os.makedirs(checkpoint_output_dir)
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(checkpoint_output_dir)
            logger.info("Saving final model checkpoint to %s", checkpoint_output_dir)
    
    return completed_steps, train_loss / completed_steps if completed_steps > 0 else 0.0


def evaluate(args, accelerator, dataloader, eval_set, model, checkpoint, has_labels=True, write_to_file=True):
    """Evaluate a model checkpoint on the given evaluation data."""

    num_examples = args.num_examples.get(eval_set, 0)
    eval_metric = None
    completed_steps = 0
    eval_loss = 0.0
    all_predictions = None
    all_references = None
    all_probabilities = None

    if has_labels:
        # Get the metric function
        try:
            eval_metric = load_metric(args.eval_metric)
        except:
            logger.warning(f"Could not load metric {args.eval_metric}, using accuracy as fallback")
            eval_metric = load_metric("accuracy")

    eval_results = {}
    model.eval()
    for _, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        eval_loss += outputs.loss.item()
        logits = outputs.logits
        
        if args.is_regression:
            predictions = logits.squeeze()
        else:
            predictions = logits.argmax(dim=-1)
            
        predictions = accelerator.gather(predictions)

        if all_predictions is None:
            all_predictions = predictions.detach().cpu().numpy()
        else:
            all_predictions = np.append(all_predictions, predictions.detach().cpu().numpy(), axis=0)

        if not args.is_regression and not hasattr(args, 'skip_probabilities'):
            probabilities = torch.nn.functional.softmax(logits, dim=-1).max(dim=-1).values
            probabilities = accelerator.gather(probabilities)
            if all_probabilities is None:
                all_probabilities = probabilities.detach().cpu().numpy()
            else:
                all_probabilities = np.append(all_probabilities, probabilities.detach().cpu().numpy(), axis=0)

        if has_labels:
            references = batch["labels"]
            references = accelerator.gather(references)
            if all_references is None:
                all_references = references.detach().cpu().numpy()
            else:
                all_references = np.append(all_references, references.detach().cpu().numpy(), axis=0)

            if eval_metric is not None:
                eval_metric.add_batch(
                    predictions=predictions,
                    references=references,
                )
        completed_steps += 1

    if has_labels and eval_metric is not None:
        try:
            eval_results.update(eval_metric.compute())
        except Exception as e:
            logger.warning(f"Error computing metric: {e}")
            eval_results[args.eval_metric] = 0.0
        
        eval_results["completed_steps"] = completed_steps
        eval_results["avg_eval_loss"] = eval_loss / completed_steps if completed_steps > 0 else 0.0

        if write_to_file:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                results_file = os.path.join(args.output_dir, f"{eval_set}_results_{checkpoint}.json")
                with open(results_file, "w") as f:
                    json.dump(eval_results, f, indent=4, sort_keys=True)

    if write_to_file:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and num_examples > 0:
            output_file = os.path.join(args.output_dir, f"{eval_set}_output_{checkpoint}.csv")
            if not args.is_regression and all_probabilities is not None and len(all_predictions) == len(all_probabilities):
                df = pd.DataFrame(list(zip(all_predictions, all_probabilities)), columns=["prediction", "probability"])
            else:
                df = pd.DataFrame(all_predictions, columns=["prediction"])
            
            # Ensure we don't exceed the actual number of examples
            if len(df) > num_examples:
                df = df.head(num_examples)
            df.to_csv(output_file, header=True, index=False)
    
    return eval_results


def load_from_pretrained(args, pretrained_model_name_or_path):
    """Load the pretrained model and tokenizer."""

    # In distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently perform this procedure.

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=args.num_labels if hasattr(args, "num_labels") else None,
        finetuning_task=args.task_name.lower() if args.task_name else None,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, use_fast=args.use_fast_tokenizer, cache_dir=args.cache_dir
    )
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            from_tf=bool(".ckpt" in pretrained_model_name_or_path),
            config=config,
            ignore_mismatched_sizes=True,
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.warning(f"Error loading model with ignore_mismatched_sizes=True: {e}")
        # Try without ignore_mismatched_sizes
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            from_tf=bool(".ckpt" in pretrained_model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    
    return config, tokenizer, model


def finetune(accelerator, model_name_or_path, train_file, output_dir, **kwargs):
    """Fine-tuning a pre-trained model on a downstream task.

    Args:
      accelerator: An instance of an accelerator for distributed training (on
        multi-GPU, TPU) or mixed precision training.
      model_name_or_path: Path to pretrained model or model identifier from
        huggingface.co/models.
      train_file: A csv or a json file containing the training data.
      output_dir: The output directory where the model predictions and checkpoints
        will be written.
      **kwargs: Dictionary of key/value pairs with which to update the
        configuration object after loading. The values in kwargs of any keys which
        are configuration attributes will be used to override the loaded values.
    """
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the
    # screen. accelerator.is_local_main_process is only True for one process per
    # machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    model_args = FTModelArguments(model_name_or_path=model_name_or_path)
    data_args = FTDataArguments(train_file=train_file)
    training_args = FTTrainingArguments(output_dir=output_dir)
    args = argparse.Namespace()

    for arg_class in (model_args, data_args, training_args):
        for key, value in vars(arg_class).items():
            setattr(args, key, value)

    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # Sanity checks
    data_files = {}
    args.data_file_extension = None

    # You need to provide the training data as we always run training
    args.do_train = True
    assert args.train_file is not None
    data_files[Split.TRAIN.value] = args.train_file

    if args.do_eval or args.eval_strategy != "no":
        assert args.eval_file is not None
        data_files[Split.EVAL.value] = args.eval_file

    if args.do_eval and args.test_file is not None:
        data_files[Split.TEST.value] = args.test_file

    if args.do_predict:
        assert args.infer_file is not None
        data_files[Split.INFER.value] = args.infer_file

    for key in data_files:
        extension = data_files[key].split(".")[-1]
        assert extension in ["csv", "json"], f"`{key}_file` should be a csv or a json file."
        if args.data_file_extension is None:
            args.data_file_extension = extension
        else:
            assert extension == args.data_file_extension, f"`{key}_file` should be a {args.data_file_extension} file`."

    try:
        available_metrics = datasets.list_metrics()
    except:
        available_metrics = ["accuracy", "precision", "recall", "f1"]
    
    assert (
        args.eval_metric in available_metrics
    ), f"{args.eval_metric} not in the list of supported metrics {available_metrics}."

    # Handle the output directory creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Loading the dataset from local csv or json files.
    try:
        raw_datasets = load_dataset(args.data_file_extension, data_files=data_files)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Labels
    is_regression = False
    if "label" in raw_datasets[Split.TRAIN.value].features:
        label_feature = raw_datasets[Split.TRAIN.value].features["label"]
        if hasattr(label_feature, 'dtype'):
            is_regression = label_feature.dtype in ["float32", "float64"]
        else:
            # Check sample values
            sample_label = raw_datasets[Split.TRAIN.value][0]["label"]
            is_regression = isinstance(sample_label, (float, np.float32, np.float64))
    
    args.is_regression = is_regression

    if args.is_regression:
        label_list = None
        num_labels = 1
    else:
        label_list = args.label_list
        if label_list is None:
            # Extract label list from dataset
            if "label" in raw_datasets[Split.TRAIN.value].features:
                if hasattr(raw_datasets[Split.TRAIN.value].features["label"], 'names'):
                    label_list = raw_datasets[Split.TRAIN.value].features["label"].names
                else:
                    # Get unique labels
                    unique_labels = set()
                    for example in raw_datasets[Split.TRAIN.value]:
                        unique_labels.add(example["label"])
                    label_list = sorted(list(unique_labels))
            else:
                raise ValueError("No label column found in dataset and no label_list provided")
        
        label_list = sorted(label_list)  # Sort for determinism
        num_labels = len(label_list)
    
    args.num_labels = num_labels

    # Load pre-trained model
    config, tokenizer, model = load_from_pretrained(args, args.model_name_or_path)

    # Preprocessing the datasets
    non_label_column_names = [name for name in raw_datasets[Split.TRAIN.value].column_names if name != "label"]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    label_to_id = None
    if not args.is_regression and label_list is not None:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None and not args.is_regression:
                # Map labels to IDs
                result["labels"] = [label_to_id.get(l, l) for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will
                # expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets[Split.TRAIN.value].column_names,
            desc="Running tokenizer on dataset",
        )

    num_examples = {}
    splits = [s.value for s in Split]
    for split in splits:
        if split in processed_datasets:
            num_examples[split] = len(processed_datasets[split])
    args.num_examples = num_examples

    train_dataset = processed_datasets[Split.TRAIN.value]
    eval_dataset = processed_datasets[Split.EVAL.value] if Split.EVAL.value in processed_datasets else None
    test_dataset = processed_datasets[Split.TEST.value] if Split.TEST.value in processed_datasets else None
    infer_dataset = processed_datasets[Split.INFER.value] if Split.INFER.value in processed_datasets else None

    # Log a few random samples from the training set:
    if len(train_dataset) > 0:
        for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
            logger.info("Sample %d of the training set: %s.", index, train_dataset[index])

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data
        # collator that will just convert everything to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by
        # padding to the maximum length of the samples passed). When using mixed
        # precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple of
        # 8s, which will enable the use of Tensor Cores on NVIDIA hardware with
        # compute capability >= 7.5 (Volta).
        # For fp8, we pad to multiple of 16.
        if hasattr(accelerator, 'mixed_precision') and accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif hasattr(accelerator, 'mixed_precision') and accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader, test_dataloader, infer_dataloader = None, None, None

    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator
        )

    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator
        )

    if infer_dataset is not None:
        infer_dataloader = DataLoader(
            infer_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    if test_dataloader is not None:
        test_dataloader = accelerator.prepare(test_dataloader)
    if infer_dataloader is not None:
        infer_dataloader = accelerator.prepare(infer_dataloader)

    # Note -> the training dataloader needs to be prepared before we grab its
    # length below (cause its length will be shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps == -1:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Train
    completed_steps, avg_train_loss = train(
        args, accelerator, model, tokenizer, train_dataloader, optimizer, lr_scheduler, eval_dataloader
    )
    accelerator.wait_for_everyone()
    logger.info("Training job completed: completed_steps = %d, avg_train_loss = %f", completed_steps, avg_train_loss)

    args.model_name_or_path = os.path.join(args.output_dir, "best-checkpoint")
    logger.info("Loading the best checkpoint: %s", args.model_name_or_path)
    
    if os.path.exists(args.model_name_or_path):
        config, tokenizer, model = load_from_pretrained(args, args.model_name_or_path)
        model = accelerator.prepare(model)
    else:
        logger.warning(f"Best checkpoint not found at {args.model_name_or_path}, using current model")

    if args.do_eval:
        # Evaluate
        if eval_dataloader is not None:
            logger.info("***** Running evaluation on the eval data using the best checkpoint *****")
            eval_results = evaluate(args, accelerator, eval_dataloader, Split.EVAL.value, model, "best-checkpoint")
            if eval_results:
                avg_eval_loss = eval_results.get("avg_eval_loss", 0.0)
                eval_metric_value = eval_results.get(args.eval_metric, 0.0)
                logger.info("Evaluation job completed: avg_eval_loss = %f", avg_eval_loss)
                logger.info("Evaluation result for the best checkpoint: %s = %f", args.eval_metric, eval_metric_value)

        if test_dataloader is not None:
            logger.info("***** Running evaluation on the test data using the best checkpoint *****")
            eval_results = evaluate(args, accelerator, test_dataloader, Split.TEST.value, model, "best-checkpoint")
            if eval_results:
                avg_eval_loss = eval_results.get("avg_eval_loss", 0.0)
                eval_metric_value = eval_results.get(args.eval_metric, 0.0)
                logger.info("Test job completed: avg_test_loss = %f", avg_eval_loss)
                logger.info("Test result for the best checkpoint: %s = %f", args.eval_metric, eval_metric_value)

    if args.do_predict:
        # Predict
        if infer_dataloader is not None:
            logger.info("***** Running inference using the best checkpoint *****")
            evaluate(
                args, accelerator, infer_dataloader, Split.INFER.value, model, "best-checkpoint", has_labels=False
            )
            logger.info("Inference job completed.")

    # Release all references to the internal objects stored and call the garbage
    # collector. You should call this method between two trainings with different
    # models/optimizers.
    accelerator.free_memory()
