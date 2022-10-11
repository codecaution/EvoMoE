# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.modules.moe import MOELayer

@register_criterion("masked_lm_gumbel")
class MaskedLmGumbelLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        logits = model(**sample["net_input"], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        moe_metadata = self.get_moe_metadata(model)
        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        logging_output.update(moe_metadata)
        return loss, sample_size, logging_output

    def get_moe_metadata(self, model):
        moe_logging_output = {}
        for key in ['gumbel_choosed_experts', ]:
            total_val = 0
            count = 0
            for _, module in model.named_modules():
                if isinstance(module, MOELayer):
                    total_val += module.metadata[key] if key in module.metadata else 0
                    count += 1
            moe_logging_output[key] = total_val / count
        moe_logging_output["batch_count"] = 1
        return moe_logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ntokens =  sum(log.get("ntokens", 0) for log in logging_outputs)
        gumbel_choosed_experts = sum(log.get("gumbel_choosed_experts", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        metrics.log_scalar(
            "gumbel_avg_experts", gumbel_choosed_experts / ntokens, ntokens, round=3
        )
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
