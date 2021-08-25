# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from fairseq import registry
from fairseq.optim.heat_scheduler.fairseq_heat_scheduler import (  # noqa
    FairseqHeatScheduler,
    LegacyFairseqHeatScheduler,
)
from omegaconf import DictConfig


(
    build_heat_scheduler_,
    register_heat_scheduler,
    HEAT_SCHEDULER_REGISTRY,
    HEAT_SCHEDULER_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--heat-scheduler", base_class=FairseqHeatScheduler, default="fixed"
)


def build_heat_scheduler(cfg: DictConfig):
    return build_heat_scheduler_(cfg)


# automatically import any Python files in the optim/heat_scheduler/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.optim.heat_scheduler." + file_name)
