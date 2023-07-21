import argparse
import json
import logging
import os
import random
import sys

import torch
from entity.fusebert import EntityModel, FuseBertModel
from entity.utils import NpEncoder, batchify, convert_dataset_to_samples
from shared.const import get_binary_labelmap, task_ner_labels
from shared.data_structures import Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("root")


def check_and_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_model(model, output_dir):
    """
    Save the model to the output directory
    """
    logger.info("Saving model to %s", (output_dir))
    torch.save(model.model, os.path.join(output_dir, "model.pth"))
    model.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    model.auxiliary_tokenizer.save_pretrained(os.path.join(output_dir, "auxiliary_tokenizer"))


def init_model(args, num_ner_labels):
    logger.info("Initializing model...")
    # Using a HuggingFace online model or a downloaded model
    logger.info("Loading Bert from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    bert = AutoModel.from_pretrained(args.model_name_or_path)
    logger.info("Loading auxiliary Bert from %s", args.auxiliary_model_name_or_path)
    auxiliary_tokenizer = BertTokenizer.from_pretrained(args.auxiliary_model_name_or_path)
    auxiliary_bert = BertModel.from_pretrained(args.auxiliary_model_name_or_path)

    model = FuseBertModel(bert, auxiliary_bert, num_ner_labels=num_ner_labels, last_hidden_dim=150, width_embedding_dim=150, max_span_length=args.max_span_length)

    ent_model = EntityModel(tokenizer, auxiliary_tokenizer, model)
    return ent_model


def load_model(model_dir):
    logger.info("Loading model from %s", model_dir)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    auxiliary_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "auxiliary_tokenizer"))
    model = torch.load(os.path.join(model_dir, "model.pth"))
    ent_model = EntityModel(tokenizer, auxiliary_tokenizer, model)
    return ent_model


def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    # span_hidden_table = {}
    tot_pred_ett = 0
    for i, curr_batch in enumerate(batches):
        output_dict = model.run_batch(curr_batch, training=False)
        pred_ner = output_dict["pred_ner"]
        for sample, preds in zip(curr_batch, pred_ner):
            off = sample["sent_start_in_doc"] - sample["sent_start"]
            k = sample["doc_key"] + "-" + str(sample["sentence_ix"])
            ner_result[k] = []
            for span, pred in zip(sample["spans"], preds):
                pred = pred.item()
                if pred == 0:
                    continue
                ner_result[k].append([span[0] + off, span[1] + off, ner_id2label[pred]])
            tot_pred_ett += len(ner_result[k])

    logger.info("Total pred entities: %d", tot_pred_ett)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc["doc_key"] + "-" + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info("%s not in NER results!", k)
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    logger.info("Output predictions to %s..", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(doc, cls=NpEncoder) for doc in js))


def evaluate(model, batches, tot_gold):
    """
    Evaluate the entity model
    """
    logger.info("Evaluating...")
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0

    for _, curr_batch in enumerate(batches):
        output_dict = model.run_batch(curr_batch, training=False)
        pred_ner = output_dict["pred_ner"]
        for sample, preds in zip(curr_batch, pred_ner):
            for gold, pred in zip(sample["spans_label"], preds):
                l_tot += 1
                if pred == gold:  # be like: array([True])
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1

    acc = l_cor / l_tot
    logger.info("Accuracy (including 0): %5f", acc)
    logger.info("Cor: %d, Pred TOT: %d, Gold TOT: %d", cor, tot_pred, tot_gold)
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info("P: %.5f, R: %.5f, F1: %.5f", p, r, f1)
    return f1


def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="if do train, the model will be initialized from --model_name_or_path and --auxiliary_model_name_or_path")
    parser.add_argument("--train_shuffle", action="store_true", help="use this to train with randomly shuffled data")
    parser.add_argument("--do_eval", action="store_true", help="if do eval, the model will be loaded from --model_dir")
    parser.add_argument("--eval_test", action="store_true", help="use this to evaluate on test set; otherwise, on dev set")

    parser.add_argument("--task", type=str, default="scierc", required=True, choices=["ace04", "ace05", "scierc"])
    parser.add_argument("--data_dir", type=str, required=True, help="path to the preprocessed dataset")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/cxr_graph/outputs/entity_model_default", help="output directory of the entity model")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/cxr_graph/models/entity_model_default", help="output directory of the entity model")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="a huggingface model repo [bert-base-uncased | allenai/scibert_scivocab_uncased | albert-xxlarge-v1] or an offline model (check https://huggingface.co/docs/transformers/installation#offline-mode for details)",
    )
    parser.add_argument(
        "--auxiliary_model_name_or_path",
        type=str,
        help="a huggingface model repo [bert-base-uncased | allenai/scibert_scivocab_uncased | albert-xxlarge-v1] or an offline model (check https://huggingface.co/docs/transformers/installation#offline-mode for details)",
    )

    parser.add_argument("--dev_pred_filename", type=str, default="ent_pred_dev.json", help="the prediction filename for the dev set")
    parser.add_argument("--test_pred_filename", type=str, default="ent_pred_test.json", help="the prediction filename for the test set")

    parser.add_argument("--context_window", type=int, default=300, required=True, help="the context window size W for the entity model")
    parser.add_argument("--max_span_length", type=int, default=8, help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument("--train_batch_size", type=int, default=16, help="batch size during training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="batch size during inference")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate for the BERT encoder")
    parser.add_argument("--task_learning_rate", type=float, default=1e-4, help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="the ratio of the warmup steps to the total steps")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of the training epochs")
    parser.add_argument("--print_loss_step", type=int, default=100, help="how often logging the loss value during training")
    parser.add_argument("--n_eval_per_epoch", type=int, default=1, help="how often evaluating the trained model on dev set during training")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    args.train_data = os.path.join(args.data_dir, "train.json")
    args.dev_data = os.path.join(args.data_dir, "dev.json")
    args.test_data = os.path.join(args.data_dir, "test.json")

    if "albert" in args.model_name_or_path:
        logger.info("Use Albert: %s", args.model_name_or_path)
        args.use_albert = True

    setseed(args.seed)
    check_and_mkdir(args.output_dir)
    check_and_mkdir(args.model_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), "w"))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), "w"))

    logger.info(sys.argv)
    logger.info(args)

    # ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    ner_label2id, ner_id2label = get_binary_labelmap(task_ner_labels[args.task])
    num_ner_labels = 1  # binary classification

    if args.do_train:
        model = init_model(args, num_ner_labels)

        train_data = Dataset(args.train_data)
        train_samples, train_ner = convert_dataset_to_samples(train_data, args.max_span_length, ner_label2id=ner_label2id, context_window=args.context_window)
        train_batches = batchify(train_samples, args.train_batch_size)
        dev_data = Dataset(args.dev_data)
        dev_samples, dev_ner = convert_dataset_to_samples(dev_data, args.max_span_length, ner_label2id=ner_label2id, context_window=args.context_window)
        dev_batches = batchify(dev_samples, args.eval_batch_size)

        best_result = 0.0

        param_optimizer = list(model.model.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer if "bert" in n]}, {"params": [p for n, p in param_optimizer if "bert" not in n], "lr": args.task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=not (args.bertadam))
        t_total = len(train_batches) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * args.warmup_proportion), t_total)

        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // args.n_eval_per_epoch
        for curr_epoch in tqdm(range(args.num_epoch)):
            if args.train_shuffle:
                random.shuffle(train_batches)
            for i in tqdm(range(len(train_batches))):
                output_dict = model.run_batch(train_batches[i], training=True)
                loss = output_dict["ner_loss"]
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:
                    logger.info("Epoch=%d, iter=%d, loss=%.5f", curr_epoch, i, tr_loss / tr_examples)
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:
                    f1 = evaluate(model, dev_batches, dev_ner)
                    if f1 > best_result:
                        best_result = f1
                        logger.info("(Epoch=%d) !!! Achieved best valid f1: %.3f", curr_epoch, f1 * 100)
                        save_model(model, args.model_dir)
                    else:
                        logger.info("(Epoch=%d) Current dev f1: %.3f, best dev f1 = %.3f", curr_epoch, f1 * 100, best_result * 100)

    if args.do_eval:
        model = load_model(args.model_dir)

        if args.eval_test:
            test_data = Dataset(args.test_data)
            prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        else:
            test_data = Dataset(args.dev_data)
            prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
        test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id, context_window=args.context_window)
        test_batches = batchify(test_samples, args.eval_batch_size)
        evaluate(model, test_batches, test_ner)
        output_ner_predictions(model, test_batches, test_data, output_file=prediction_file)
