import os
import torch
from transformers import Seq2SeqTrainer


class MonoT5Trainer(Seq2SeqTrainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder_input_ids = None
        self.token_false_id = self.tokenizer.get_vocab()["▁false"]
        self.token_true_id = self.tokenizer.get_vocab()["▁true"]

        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        if "inst_w" in inputs.keys():
            inst_w = inputs.pop("inst_w")
        if self.decoder_input_ids is None:
            if isinstance(model, torch.nn.DataParallel) or isinstance(
                model, torch.nn.parallel.DistributedDataParallel
            ):
                self.decoder_input_ids = model.module._shift_right(inputs["labels"])
            else:
                self.decoder_input_ids = model._shift_right(inputs["labels"])
        if self.loss_func == "cross_entropy":
            if isinstance(model, torch.nn.DataParallel) or isinstance(
                model, torch.nn.parallel.DistributedDataParallel
            ):
                inputs["decoder_input_ids"] = model.module._shift_right(
                    inputs["labels"]
                )
            else:
                inputs["decoder_input_ids"] = model._shift_right(inputs["labels"])
            return super().compute_loss(model, inputs, return_outputs)
        elif self.loss_func in ["contrastive", "ensemble"]:
            xe_loss, logits = model(**inputs, use_cache=False)[:2]
            logits = logits[:, -1, [self.token_false_id, self.token_true_id]]
            scores = torch.nn.functional.log_softmax(logits, dim=1)
            log_probs = scores[:, 1]
            loss = torch.mean(
                -torch.log(
                    torch.exp(log_probs[0]) / torch.sum(torch.exp(log_probs), dim=-1)
                )
            )
        elif self.loss_func == "weighted_cross_entropy":
            xe_loss, logits = model(**inputs, use_cache=False)[:2]
            loss = inst_w * torch.nn.CrossEntropyLoss(
                ignore_index=-100, reduction="none"
            )(logits.view(-1, logits.size(-1)), inputs["labels"].view(-1))
            loss = torch.mean(loss)
        else:
            raise ValueError(self.loss_func)

        return loss
