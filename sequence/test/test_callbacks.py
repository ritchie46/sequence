from sequence import callbacks
from dumpster.registries.file import ModelRegistry


def test_register_global_step():
    mr = ModelRegistry("test")
    f = callbacks.register_global_step(mr)
    f(global_step=13,)
    assert mr.global_step_ == 13
