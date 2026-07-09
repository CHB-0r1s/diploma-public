"""Тестовая настройка: делаем `ifd_select` импортируемым и подменяем torch заглушкой.

Модуль `ifd_select` лежит в notebooks/ и на верхнем уровне делает `import torch`.
Юнит-тесты покрывают только чистую логику (select / parsing / cache / strip_context),
которой настоящий torch не нужен, поэтому в CI мы не тянем тяжёлый пакет — вместо него
кладём минимальную заглушку в sys.modules ДО импорта ifd_select.
Если настоящий torch установлен, заглушка не используется.
"""

import os
import sys
import types

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    try:  # настоящий torch есть — ничего не подменяем
        import torch  # noqa: F401

        return
    except ImportError:
        pass

    class _NoGrad:
        def __call__(self, fn):
            return fn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch = types.ModuleType("torch")
    torch.no_grad = lambda: _NoGrad()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    functional.log_softmax = lambda *a, **k: None
    nn.functional = functional
    torch.nn = nn

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional


_install_torch_stub()
