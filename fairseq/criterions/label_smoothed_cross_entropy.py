# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion, MoECriterion
from fairseq.dataclass import FairseqDataclass
from fairseq.modules.moe import MOELayer
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True



@dataclass
class MoELabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    moe_gate_loss_wt: float = field(
        default=1.0,
        metadata={
            "help": "Weight associated with MoE gate loss"
                    "in the weighted sum of gate loss and cross entropy loss"
        }
    )
    moe_gate_loss_combine_method: str = field(
        default="average",
        metadata={
            "help": "Method of combining the gate loss from each MoE layers"
                    "('sum', 'average')"
        }
    )
    moe_gate_loss_transform: str = field(
        default="none",
        metadata={
            "help": "Transformation to apply to the gate loss ('none', 'neg_log')"
        }
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )

@register_criterion("moe_label_smoothed_cross_entropy", dataclass=MoELabelSmoothedCrossEntropyCriterionConfig)
class MoELabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    moe_logging_keys = [
        "overflow_expert1",        # average % of overflowed tokens from 1st expert
        "overflow_expert2",        # average % of overflowed tokens from 2nd expert
        "entropy_gating",          # average entropy of the gating distribution
        "expert1_balance_top",     # average cumulative % of tokens processed by the most used 20% 1st experts
        "expert1_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 1st experts
        "unused_expert1_count",    # average number of 1st experts which process no tokens
        "expert2_balance_top",     # average cumulative % of tokens processed by the most used 20% 2nd experts
        "expert2_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 2nd experts
        "unused_expert2_count",    # average number of 2nd experts which process no tokens
        "all_to_all_cpu_time_ms",  # CPU time spent in all to all calls in milliseconds
        "all_to_all_cuda_time_ms", # CUDA ttime spent in all to all calls in milliseconds
    ]

    def __init__(self, task, moe_gate_loss_wt, moe_gate_loss_combine_method, moe_gate_loss_transform, sentence_avg, \
                        label_smoothing, report_accuracy, ignore_prefix_size):
        super().__init__(task)
        self.gate_loss_weight = moe_gate_loss_wt
        self.gate_loss_combine_method = moe_gate_loss_combine_method
        self.gate_loss_transform = moe_gate_loss_transform
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.report_accuracy = report_accuracy
        self.ignore_prefix_size = ignore_prefix_size

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, inner_loss, moe_loss, moe_metadata, sample_size, logging_output = self.compute_loss(model, sample, reduce=reduce)
        logging_output["loss"] = loss.data
        logging_output["moe_loss"] = moe_loss.data
        logging_output.update(moe_metadata)

        return loss, sample_size, logging_output


    def compute_loss(self, model, sample, reduce=True):
        net_output, inner_loss, sample_size, logging_output = self.compute_inner_loss(model, sample)
        # gate_loss = 0.0
        gate_loss = torch.Tensor([0.0,]).cuda()
        gate_count = 0
        for l_aux in net_output[1]["l_aux"]:
            if l_aux is not None:
                gate_loss += l_aux
                gate_count += 1
        gate_count = 1 if gate_count == 0 else gate_count
        if self.gate_loss_combine_method == "average":
            gate_loss = gate_loss / gate_count
        if self.gate_loss_transform == "neg_log":
            gate_loss = - torch.log(gate_loss)
        gate_loss = sample_size * gate_loss
        loss = inner_loss + self.gate_loss_weight * gate_loss
        return loss, inner_loss, gate_loss, self.get_moe_metadata(model), sample_size, logging_output


    def compute_inner_loss(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )        
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        return net_output, loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def get_moe_metadata(self, model):
        moe_logging_output = {}
        for key in MoECriterion.moe_logging_keys + ['gumbel_choosed_experts', ]:
            total_val = 0
            count = 0
            for _, module in model.named_modules():
                if isinstance(module, MOELayer):
                    total_val += module.metadata[key] if key in module.metadata else 0
                    count += 1
            count = 1 if count == 0 else count
            moe_logging_output[key] = total_val / count
        moe_logging_output["batch_count"] = 1
        return moe_logging_output

    @staticmethod
    def reduce_moe_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        moe_loss_sum = sum(log.get("moe_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "moe_gate_loss", moe_loss_sum / sample_size, sample_size, round=8
        )
        gumbel_choosed_experts = sum(log.get("gumbel_choosed_experts", 0) for log in logging_outputs)
        metrics.log_scalar(
                "gumbel_avg_experts", gumbel_choosed_experts / sample_size, sample_size, round=3
        )
        batch_count = sum(log.get("batch_count", 0) for log in logging_outputs)
        for key in MoECriterion.moe_logging_keys:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(
                key, val / batch_count, batch_count, round=3
            )

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        MoELabelSmoothedCrossEntropyCriterion.reduce_moe_metrics(logging_outputs)

        loss_sum = sum(log.get("inner_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "inner_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True