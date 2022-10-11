# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.heat_scheduler import FairseqHeatScheduler, register_heat_scheduler


@dataclass
class FixedheatScheduleConfig(FairseqDataclass):
    force_anneal: Optional[int] = field(
        default=None,
        metadata={"help": "force annealing at specified epoch"},
    )
    heat_shrink: float = field(
        default=0.1,
        metadata={"help": "shrink factor for annealing, heat_new = (heat * heat_shrink)"},
    )
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    heat: List[float] = II("optimization.heat")


@register_heat_scheduler("fixed", dataclass=FixedheatScheduleConfig)
class FixedheatSchedule(FairseqHeatScheduler):
    """Decay the heat on a fixed schedule."""

    def __init__(self, cfg: FixedheatScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)

        self.heat = cfg.heat[0]
        if cfg.warmup_updates > 0:
            self.warmup_factor = 1.0 / cfg.warmup_updates
        else:
            self.warmup_factor = 1

    def state_dict(self):
        return {"heat": self.heat}

    def load_state_dict(self, state_dict):
        if "heat" in state_dict:
            self.heat = state_dict["heat"]

    def get_next_heat(self, epoch):
        heats = self.cfg.heat
        if self.cfg.force_anneal is None or epoch < self.cfg.force_anneal:
            # use fixed heat schedule
            next_heat = heats[min(epoch - 1, len(heats) - 1)]
        else:
            # annneal based on heat_shrink
            next_heat = heats[-1] * self.cfg.heat_shrink ** (
                epoch + 1 - self.cfg.force_anneal
            )
        return next_heat

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.heat = self.get_next_heat(epoch)
        self.optimizer.set_heat(self.warmup_factor * self.heat)
        return self.optimizer.get_heat()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.cfg.warmup_updates > 0 and num_updates < self.cfg.warmup_updates:
            self.warmup_factor = (num_updates + 1) / float(self.cfg.warmup_updates)
            self.optimizer.set_heat(self.warmup_factor * self.heat)
        else:
            self.optimizer.set_heat(self.heat)
        return self.optimizer.get_heat()
