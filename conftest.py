# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Put the repo root on ``sys.path`` for the test session.

``tests/`` is not a package, so pytest inserts ``tests/`` -- not the repo root -- onto ``sys.path``.
``ptx_anneal`` is found anyway because CI pip-installs it, but ``e2e/`` is deliberately never
packaged, so without this the triton-native tests could only ever skip.

They still skip when ``e2e/`` genuinely is not there (an sdist, which excludes it) -- see the
``importorskip`` in ``tests/test_triton_native.py``. This file only makes the checkout's own layout
importable; it does not make an absent directory appear.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
