# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.heat_scheduler import FairseqHeatScheduler, register_heat_scheduler


@dataclass
class CosineheatScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the Heat linearly for the first N updates"},
    )
    warmup_init_heat: float = field(
        default=-1,
        metadata={
            "help": "initial Heat during warmup phase; default is cfg.heat"
        },
    )
    heat: List[float] = field(
        default=II("optimization.heat"),
        metadata={"help": "max learning rate, must be more than cfg.min_heat"},
    )
    min_heat: float = field(default=0.0, metadata={"help": "min heat"})
    t_mult: float = field(
        default=1.0, metadata={"help": "factor to grow the length of each period"}
    )
    heat_period_updates: float = field(
        default=-1, metadata={"help": "initial number of updates per period"}
    )
    heat_shrink: float = field(
        default=0.1, metadata={"help": "shrink factor for annealing"}
    )
    # This is not required, but is for convenience in inferring heat_period_updates
    max_update: int = II("optimization.max_update")


@register_heat_scheduler("cosine", dataclass=CosineheatScheduleConfig)
class CosineheatSchedule(FairseqHeatScheduler):
    """Assign heat based on a cyclical schedule that follows the cosine function.

    See https://arxiv.org/pdf/1608.03983.pdf for details.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-heat``) until the configured
    max learning rate (``--heat``).

    During warmup::

      heats = torch.linspace(cfg.warmup_init_heat, cfg.heat, cfg.warmup_updates)
      heat = heats[update_num]

    After warmup::

      heat = cfg.min_heat + 0.5*(cfg.heat - cfg.min_heat)*(1 + cos(t_curr / t_i))

    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(self, cfg: CosineheatScheduleConfig, fairseq_optimizer):
        super().__init__(cfg, fairseq_optimizer)
        if isinstance(cfg.heat, Collection) and len(cfg.heat) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with cosine."
                f" Consider --heat-scheduler=fixed instead. ({cfg.heat})"
            )

        self.max_heat = cfg.heat[0] if isinstance(cfg.heat, Collection) else cfg.heat
        assert (
            self.max_heat > cfg.min_heat
        ), f"max_heat (={cfg.heat}) must be more than min_heat (={cfg.min_heat})"

        warmup_end_heat = self.max_heat
        if cfg.warmup_init_heat < 0:
            cfg.warmup_init_heat = cfg.min_heat

        self.t_mult = cfg.t_mult
        self.period = cfg.heat_period_updates

        if self.period <= 0:
            assert (
                cfg.max_update > 0
            ), "Either --max_update or --heat-period-updates must be set"
            self.period = cfg.max_update - cfg.warmup_updates

        if cfg.warmup_updates > 0:
            # linearly warmup for the first cfg.warmup_updates
            self.heat_step = (warmup_end_heat - cfg.warmup_init_heat) / cfg.warmup_updates
        else:
            self.heat_step = 1

        self.warmup_updates = cfg.warmup_updates
        self.heat_shrink = cfg.heat_shrink

        # initial learning rate
        self.heat = cfg.warmup_init_heat
        self.optimizer.set_heat(self.heat)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_heat()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.cfg.warmup_updates:
            self.heat = self.cfg.warmup_init_heat + num_updates * self.heat_step
        else:
            curr_updates = num_updates - self.cfg.warmup_updates
            if self.t_mult != 1:
                i = math.floor(
                    math.log(
                        1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult
                    )
                )
                t_i = self.t_mult ** i * self.period
                t_curr = (
                    curr_updates
                    - (1 - self.t_mult ** i) / (1 - self.t_mult) * self.period
                )
            else:
                i = math.floor(curr_updates / self.period)
                t_i = self.period
                t_curr = curr_updates - (self.period * i)

            heat_shrink = self.heat_shrink ** i
            min_heat = self.cfg.min_heat * heat_shrink
            max_heat = self.max_heat * heat_shrink

            self.heat = min_heat + 0.5 * (max_heat - min_heat) * (
                1 + math.cos(math.pi * t_curr / t_i)
            )

        self.optimizer.set_heat(self.heat)
        return self.heat
