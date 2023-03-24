import torch
import torch.nn as nn
from transformers import PreTrainedModel
import torch.nn.functional as F



class BiEncoder(PreTrainedModel):
    def __init__(self, cont_config, cand_config, *inputs, **kwargs):
        super().__init__(cont_config, cand_config, *inputs, **kwargs)
        # if shared is true it creates only one model (Siamese type)
        if kwargs["shared"] is True:
            self.cont_bert = kwargs["cont_bert"]
            self.cand_bert = self.cont_bert
        else:
            self.cont_bert = kwargs["cont_bert"]
            self.cand_bert = kwargs["cand_bert"]
        self.input_size = None
        self.l_type = kwargs["loss_type"]
        self.l_func = kwargs["loss_function"]

    def loss_func(self, candidate_vec, context_vec, labels, l_type="cos", l_func="MSE"):
        candidate_vec = candidate_vec.squeeze(1)
        labels = labels.squeeze(1)
        if l_type == "dot":
            if l_func == "contrastive":
                dot_product = torch.matmul(context_vec, candidate_vec.t())
                mask = torch.eye(self.input_size).to(candidate_vec.device)
                dot_product = 1 - F.log_softmax(dot_product, dim=-1) * mask
                dot_product = dot_product.mean(dim=-1)
                loss = 0.5 * (
                    labels.float() * dot_product.pow(2)
                    + (1 - labels).float() * F.relu(0.5 - dot_product).pow(2)
                )
                loss = loss.mean()
            else:
                dot_product = torch.mul(context_vec, candidate_vec)
                loss_fnct = nn.MSELoss()
                loss = loss_fnct(dot_product.mean(dim=-1), target=labels)
        else:
            cos = nn.CosineSimilarity()
            if l_func == "contrastive":
                loss_angle = 1 - cos(context_vec, candidate_vec)
                loss = 0.5 * (
                    labels.float() * loss_angle.pow(2)
                    + (1 - labels).float() * F.relu(0.5 - loss_angle).pow(2)
                )
                loss = loss.mean(dim=-1)
            else:
                loss_angle = cos(context_vec, candidate_vec)
                loss_fnct = nn.MSELoss()
                loss = loss_fnct(loss_angle, target=labels)
        return loss

    def forward(
        self,
        context_input_ids=None,
        context_input_masks=None,
        candidate_input_ids=None,
        candidate_input_masks=None,
        labels=None,
        get_embedding=None,
        mode="train",
        pooling="mean",
    ):
        # only select the first candidate (whose lbl==1)
        # if labels is not None:
        #   candidate_input_ids = candidate_input_ids[:, 0, :].unsqueeze(1)
        #   candidate_input_masks = candidate_input_masks[:, 0, :].unsqueeze(1)
        # gets the context embedding
        if get_embedding == "context" or mode == "train" or mode == "eval":
            self.input_size = context_input_ids.size(0)
            context_vec = self.cont_bert(context_input_ids, context_input_masks)[0]
            if pooling == "mean":
                # Mean pooling
                output_vectors = []
                input_mask_expanded = (
                    context_input_masks.unsqueeze(-1).expand(context_vec.size()).float()
                )
                sum_embeddings = torch.sum(context_vec * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                output_vectors.append(sum_embeddings / sum_mask)
                context_vec = torch.cat(output_vectors, 1)
                if get_embedding == "context":
                    return context_vec
        # gets the candidate embedding
        if get_embedding == "candidate" or mode == "train" or mode == "eval":
            batch_size, res_cnt, seq_length = candidate_input_ids.shape
            candidate_input_ids = candidate_input_ids.view(-1, seq_length)
            candidate_input_masks = candidate_input_masks.view(-1, seq_length)
            candidate_vec = self.cand_bert(candidate_input_ids, candidate_input_masks)[
                0
            ]
            if pooling == "mean":
                # Mean pooling
                output_vectors = []
                input_mask_expanded = (
                    candidate_input_masks.unsqueeze(-1)
                    .expand(candidate_vec.size())
                    .float()
                )
                sum_embeddings = torch.sum(candidate_vec * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                output_vectors.append(sum_embeddings / sum_mask)
                candidate_vec = torch.cat(output_vectors, 1)
                candidate_vec = candidate_vec.view(batch_size, res_cnt, -1)
                if get_embedding == "candidate":
                    return candidate_vec
        if labels is not None and mode == "train":
            return self.loss_func(
                candidate_vec=candidate_vec,
                context_vec=context_vec,
                labels=labels,
                l_type=self.l_type,
                l_func=self.l_func,
            )
