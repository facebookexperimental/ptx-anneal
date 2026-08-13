# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Engine adapters: standalone scripts run in the *engine's* interpreter.

An adapter is the bring-your-own-engine boundary. It imports ONLY its engine (+ stdlib) -- never
``ptx_anneal`` -- so it can run under an interpreter that has the engine but not this package (and so
the harness can be a frozen, engine-free binary). It drives the engine's search, scoring each
candidate by shelling out to the harness ``_score`` subcommand, and writes the winner to a result
file. See :mod:`ptx_anneal.adapters.compileiq_adapter` for the contract.
"""
