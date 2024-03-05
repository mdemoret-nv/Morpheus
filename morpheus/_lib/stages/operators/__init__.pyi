from __future__ import annotations
import morpheus._lib.stages.operators
import typing
import mrc.core.segment

__all__ = [
    "ControlMessageRouter",
    "ControlMessageWritableAcceptor",
    "ControlMessageWritableProvider"
]


class ControlMessageRouter(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str) -> None: ...
    def get_source(self, arg0: object) -> ControlMessageWritableAcceptor: ...
    pass
class ControlMessageWritableAcceptor():
    pass
class ControlMessageWritableProvider():
    pass
