import logging

import torch
from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
from torch import nn

logger = logging.getLogger("root")


class FuseBertModel(nn.Module):
    def __init__(self, bert, auxiliary_bert, num_ner_labels=2, last_hidden_dim=150, width_embedding_dim=150, max_span_length=8):
        super(FuseBertModel, self).__init__()
        # Pretrained BERT model
        self.bert = bert
        self.auxiliary_bert = auxiliary_bert
        # share the dropout layer
        self.hidden_dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(max_span_length + 1, width_embedding_dim)

        self.ner_classifier = nn.Sequential(
            FeedForward(
                input_dim=self.bert.config.hidden_size * 2 + self.auxiliary_bert.config.hidden_size * 2 + width_embedding_dim,
                num_layers=2,
                hidden_dims=[self.bert.config.hidden_size * 2, last_hidden_dim],
                activations=nn.ReLU(),
                dropout=0.2,
            ),
            nn.Linear(last_hidden_dim, num_ner_labels),
        )

    def _get_bert_embeddings(self, bert_model, input_ids, spans, attention_mask):
        # torch.Size: [batch_size, tokens_num, 768(tok_emb)]
        output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = self.hidden_dropout(output.last_hidden_state)

        # spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        # spans_start: [batch_size, spans_num(start_tok)]
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        # torch.Size: [batch_size, spans_num(start_tok), 768(tok_emb)]
        spans_start_embedding = batched_index_select(last_hidden_state, spans_start)
        # torch.Size: [batch_size, spans_num(end_tok)]
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(last_hidden_state, spans_end)

        return spans_start_embedding, spans_end_embedding

    def _get_span_embeddings(self, input_dict, auxiliary_input_dict):
        spans_start_embedding, spans_end_embedding = self._get_bert_embeddings(
            self.bert, input_ids=input_dict["tokens_tensor"], spans=input_dict["bert_spans_tensor"], attention_mask=input_dict["attention_mask_tensor"]
        )

        auxiliary_spans_start_embedding, auxiliary_spans_end_embedding = self._get_bert_embeddings(
            self.auxiliary_bert, input_ids=auxiliary_input_dict["tokens_tensor"], spans=auxiliary_input_dict["bert_spans_tensor"], attention_mask=auxiliary_input_dict["attention_mask_tensor"]
        )

        spans = input_dict["bert_spans_tensor"]
        # torch.Size([16, 652]) batch_size, spans_num(span_width)
        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        # torch.Size([16, 652, 150])
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        # spans_embedding: batch_size, spans_num, 768*4+150=3222 dims
        spans_embedding = torch.cat((spans_start_embedding, spans_end_embedding, auxiliary_spans_start_embedding, auxiliary_spans_end_embedding, spans_width_embedding), dim=-1)
        return spans_embedding

    def forward(self, input_dict, auxiliary_input_dict, eval_mode=False):
        spans_embedding = self._get_span_embeddings(input_dict, auxiliary_input_dict)
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            # torch.Size([16, 652, 3222]) -FeedForward-> torch.Size([16, 652, 150])
            # torch.Size([16, 652, 150]) -Linear-> torch.Size([16, 652, 1])
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        spans_ner_label = input_dict["spans_ner_label_tensor"]
        if eval_mode:
            return {"logits": logits, "spans_embedding": spans_embedding}
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="sum")
            active_logits = logits.view(-1, logits.shape[-1])  # (batch_size * spans_num, 1)
            active_labels = spans_ner_label.view(-1, 1).float()  # (batch_size * spans_num, 1)
            loss = loss_fct(active_logits, active_labels)
            return {"loss": loss, "logits": logits, "spans_embedding": spans_embedding}


class EntityModel:
    def __init__(self, tokenizer, auxiliary_tokenizer, model):
        super().__init__()

        self.tokenizer = tokenizer
        self.auxiliary_tokenizer = auxiliary_tokenizer

        self.model = model

        self._model_device = "cpu"
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error("No CUDA found!")
            exit(-1)
        logger.info("Moving to CUDA...")
        self._model_device = "cuda"
        self.model.cuda()
        logger.info("# GPUs = %d", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _get_input_tensors(self, tokenizer, tokens, spans, spans_ner_label):
        """convert tokens to sub-tokens then to ids according to the given tokenizer; the size of spans will not change; the index within spans will change to index of sub-tokens' position"""
        start2idx = []
        end2idx = []

        bert_tokens = []
        bert_tokens.append(tokenizer.cls_token)
        for token in tokens:
            # e.g. "trans-context-free" will be split into 5 sutokens by self.tokenizer.tokenize(),
            # supposing start2idx will append 6, then end2idx will append 10
            start2idx.append(len(bert_tokens))
            sub_tokens = tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(tokenizer.sep_token)

        indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        # span从原始token的索引，改成bert_subtoken的索引
        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, tokenizer, samples_list, model_device):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        # sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = sample["tokens"]
            spans = sample["spans"]
            spans_ner_label = sample["spans_label"]

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = self._get_input_tensors(tokenizer, tokens, spans, spans_ner_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1]
            if tokens_tensor.shape[1] > max_tokens:
                max_tokens = tokens_tensor.shape[1]
            if bert_spans_tensor.shape[1] > max_spans:
                max_spans = bert_spans_tensor.shape[1]
        #     sentence_length.append(sample["sent_length"])
        # sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list, spans_ner_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        return {
            "tokens_tensor": final_tokens_tensor.to(model_device),
            "attention_mask_tensor": final_attention_mask.to(model_device),
            "bert_spans_tensor": final_bert_spans_tensor.to(model_device),
            "spans_mask_tensor": final_spans_mask_tensor.to(model_device),
            "spans_ner_label_tensor": final_spans_ner_label_tensor.to(model_device),
        }

    def run_batch(self, samples_list, training=True):
        # Convert samples to a dict of input tensors, including the following keys:
        # tokens_tensor,  # torch.Size([16, 273]), batch_size, max_tok_num_in_batch
        # attention_mask_tensor,  # torch.Size([16, 273]), batch_size, max_tok_num_in_batch
        # bert_spans_tensor,  # torch.Size([16, 652, 3]),  batch_size, max_span_num_in_batch, 3_dims (start_subtok, end_subtok, span_width), e.g. [[1,1,1],[1,2,2],[1,7,7],...,[29,29,1],...,[0,0,0]]
        # spans_mask_tensor,  # torch.Size([16, 652]), batch_size, max_span_num_in_batch
        # spans_ner_label_tensor,  # torch.Size([16, 652]), batch_size, max_span_num_in_batch, e.g. [5,0,0,...,0,2,0,...,0]
        input_tensors_dict = self._get_input_tensors_batch(self.tokenizer, samples_list, self._model_device)
        auxiliary_input_tensors_dict = self._get_input_tensors_batch(self.auxiliary_tokenizer, samples_list, self._model_device)

        output_dict = {
            "ner_loss": 0,
        }

        if training:
            self.model.train()
            out = self.model(input_dict=input_tensors_dict, auxiliary_input_dict=auxiliary_input_tensors_dict, eval_mode=False)
            output_dict["ner_loss"] = out["loss"].sum()
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(input_dict=input_tensors_dict, auxiliary_input_dict=auxiliary_input_tensors_dict, eval_mode=True)
            # In the inference phase, instead of using the Sigmoid function to convert the outputs to probability values, we generally use the outputs of the model (often called logits) directly to make inferences
            ner_logits = out["logits"]
            threshold = 0.5
            predicted_label = (ner_logits > threshold).int()
            predicted_label = predicted_label.cpu().numpy()

            predicted = []
            pred_prob = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                for j in range(len(sample["spans"])):
                    ner.append(predicted_label[i][j])
                    prob.append(ner_logits[i][j].cpu().numpy())
                predicted.append(ner)
                pred_prob.append(prob)
            output_dict["pred_ner"] = predicted
            output_dict["ner_probs"] = pred_prob

        return output_dict
