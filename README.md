# Simulation Experiments on RadGraph

## Setup for dygie++ (deprecated)

1. Download [RadGraph](https://physionet.org/content/radgraph/1.0.0/) to `resources/radgrph` excluding the `MIMIC-CXR_graphs.json`.
2. Go to `\resources` and clone the DYGIE++ repository: run `git clone https://github.com/dwadden/dygiepp.git`.
3. Follows `radgraph\models\README.md` to setup `DYGIE++` and fo inference.

## Setup for PURE

1. Clone [PURE](https://github.com/princeton-nlp/PURE) to `resources` using `git clone https://github.com/princeton-nlp/PURE.git`.
2. Using python=3.9, delete the versions in the requirements.txt, install requirements.
3. Download data sciERC, download models.

### Install requirements

1. [√] conda create --name cxr_graph python=3.9 -y
2. [√] conda activate cxr_graph
3. [√] pip install tqdm
4. [√] pip install allennlp
   1. Install `allennlp` before `torch` as it may not catch up the latest `torch` version
5. pip install torch
   1. will be install automatically after allennlp
6. pip install transformers
   1. will be install automatically after allennlp
7. [√] pip install overrides
8. pip install requests
   1. will be install automatically after allennlp

### Download HuggingFace model (if necessary)

1. Follows the [link](https://huggingface.co/docs/transformers/installation#offline-mode)


### Issues

We are running code written in the Hugging Face transformers v3 library on the transformers v4 library

1. Allennlp does not support windowns
2. `TypeError: torch.nn.functional.relu is not a Module subclass`
   1. Replace `F.relu` to `nn.ReLU()` in `PURE\entity\models.py`
3. `ERROR - root - No CUDA found!`
   1. Comment out `self.move_model_to_cuda()` in `PURE\entity\models.py`
4. `TypeError: dropout(): argument 'input' (position 1) must be Tensor, not str`
   1. Add the augument `return_dict=False` to `BERTModel.from_pretrained(...)` in `PURE\entity\models.py`
   2. (See [link](https://stackoverflow.com/questions/65082243/dropout-argument-input-position-1-must-be-tensor-not-str-when-using-bert))
5. `TypeError: issubclass() arg 1 must be a class`
   1. `pip install typing_extensions==4.4.0`
   2. See [link](https://github.com/explosion/spaCy/issues/12659)
   3. check version: `pip show spacy`
6. `RuntimeError: CUDA error: no kernel image is available for execution on the device` 
   1. `nvcc -V` to check cuda version
   2. `python` -> `import torch` -> `print(torch.__version__)` to check torch+cuda version
   3. Find the correct torch version+cuda from [here](https://pytorch.org/get-started/previous-versions/) to install (e.g. pytorch1.12.1 + cuda11.3)
      1. `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`
      2. `pip install chardet`

## Span pruning model

This is based on the PURE model. We modified the PURE model to be a binary classification model

### Train

Exp 0, PURE-based model, CrossEntropyLoss, 2 labels:
python /root/workspace/cxr_graph/run_entity_binary.py \
   --task scierc \
   --model_name_or_path /root/workspace/hf_offline_models/scibert_scivocab_uncased \
   --data_dir /root/workspace/data/scierc_data/processed_data/json \
   --output_dir /root/workspace/cxr_graph/models/ent_scierc_scib_test_0 \
   --do_train --train_shuffle \
   --do_eval --eval_test \
   --task_learning_rate 5e-4 --context_window 300

Exp 1, Fuse-bert model, BCEWithLogitsLoss, 1 labels:
python /root/workspace/cxr_graph/run_entity_binary.py \
   --task scierc \
   --model_name_or_path /root/workspace/hf_offline_models/scibert_scivocab_uncased \
   --auxiliary_model_name_or_path /root/workspace/hf_offline_models/bert-base-uncased \
   --data_dir /root/workspace/data/scierc_data/processed_data/json \
   --output_dir /root/workspace/cxr_graph/outputs/ent_scierc_scib_test_1 \
   --model_dir /root/workspace/cxr_graph/models/ent_scierc_scib_test_1 \
   --do_train --train_shuffle \
   --do_eval --eval_test \
   --task_learning_rate 5e-4 --context_window 300

### Eval

python /root/workspace/cxr_graph/run_entity_binary.py \
   --task scierc \
   --model_name_or_path /root/workspace/hf_offline_models/scibert_scivocab_uncased \
   --data_dir /root/workspace/data/scierc_data/processed_data/json \
   --output_dir /root/workspace/cxr_graph/models/ent_scierc_scib_test_0 \
   --do_eval --eval_test \
   --task_learning_rate 5e-4 --context_window 300