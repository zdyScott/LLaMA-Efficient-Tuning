# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method.
# This code is inspired by
# https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py


from utils import (
    DynamicDataCollatorWithPadding,
    Seq2SeqPeftTrainer,
    ComputeMetrics,
    LogCallback,
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    get_logits_processor,
    plot_loss
)

import os
import debugpy

# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()
debugpy.breakpoint()
print('break on this line')

def main():

    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="sft")
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, finetuning_args, training_args.do_train, stage="sft")
    tokenizer_dataset_path = '/root/workspace_law/data_baichuan/tokenizer_dataset.hf'
    if os.path.exists(tokenizer_dataset_path):
        from datasets import load_from_disk
        dataset = load_from_disk(tokenizer_dataset_path)
    else:
        dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="sft")
        dataset.save_to_disk(tokenizer_dataset_path)
    data_collator = DynamicDataCollatorWithPadding(
        tokenizer=tokenizer,
        ignore_pad_token_for_loss=(data_args.ignore_pad_token_for_loss and not training_args.predict_with_generate)
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.eval_num_beams if \
                data_args.eval_num_beams is not None else training_args.generation_num_beams

    # Split the dataset
    if training_args.do_train:
        if data_args.dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            trainer_kwargs = {"train_dataset": dataset}
    else: # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # Initialize our Trainer
    trainer = Seq2SeqPeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **trainer_kwargs
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.7,
        "max_new_tokens": data_args.max_target_length + 1,
        "temperature": 0.95,
        "logits_processor": get_logits_processor()
    }

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
