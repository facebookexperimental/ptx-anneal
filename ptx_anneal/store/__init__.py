# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The tuning-artifact store: a content-addressed cache keyed by ``(target, arch, ir_hash)``.

:class:`~ptx_anneal.store.base.Store` is the ABC; :class:`~ptx_anneal.store.local.LocalStore`
is the filesystem implementation that ships here. Remote stores (e.g. a company artifact
service) are injected out-of-tree and implement the same ABC.
"""

from .base import Store
from .local import DEFAULT_STORE_ENV, LocalStore, default_store_root

__all__ = ["Store", "LocalStore", "default_store_root", "DEFAULT_STORE_ENV"]
