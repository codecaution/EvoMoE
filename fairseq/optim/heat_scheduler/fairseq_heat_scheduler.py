# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.optim import FairseqOptimizer


class FairseqHeatScheduler(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.best = None

    @classmethod
    def add_args(cls, parser):
        """Add arguments to the parser for this heat scheduler."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    def state_dict(self):
        """Return the Heat scheduler state dict."""
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        """Load an Heat scheduler state dict."""
        self.best = state_dict["best"]

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        pass

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        # return self.optimizer.get_heat()


class LegacyFairseqHeatScheduler(FairseqHeatScheduler):
    def __init__(self, args: Namespace):
        self.args = args
        self.best = None
