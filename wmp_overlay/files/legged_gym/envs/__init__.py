# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Overlay additions (this project): Go2 + LHR task registration.
# Added lines vs upstream are marked "SWAP-lite".

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.a1_amp_config import A1AMPCfg, A1AMPCfgPPO
from .go2.go2_amp_config import GO2AMPCfg, GO2AMPCfgPPO
from .go2.go2_amp_lhr_config import GO2AMPLHRCfg, GO2AMPLHRCfgPPO  # SWAP-lite: stage 3


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_amp", LeggedRobot, A1AMPCfg(), A1AMPCfgPPO() )
task_registry.register( "go2_amp", LeggedRobot, GO2AMPCfg(), GO2AMPCfgPPO() )
task_registry.register( "go2_amp_lhr", LeggedRobot, GO2AMPLHRCfg(), GO2AMPLHRCfgPPO() )  # SWAP-lite: R0-LHR
