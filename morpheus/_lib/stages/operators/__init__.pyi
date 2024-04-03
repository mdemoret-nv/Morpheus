from __future__ import annotations
import morpheus._lib.stages.operators
import typing
import mrc.core.common
import mrc.core.segment

__all__ = [
    "ControlMessageDynamicZip",
    "ControlMessageReadableAcceptor",
    "ControlMessageReadableProvider",
    "ControlMessageRouter",
    "ControlMessageWritableAcceptor",
    "ControlMessageWritableProvider"
]


class ControlMessageDynamicZip(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str, max_outstanding: int = 64) -> None: ...
    def get_sink(self, arg0: object) -> ControlMessageWritableProvider: ...
    pass
class ControlMessageReadableAcceptor(mrc.core.common.ReadableAcceptor):
    pass
class ControlMessageReadableProvider(mrc.core.common.ReadableProvider):
    pass
class ControlMessageRouter(mrc.core.segment.SegmentObject):
    def __init__(self, builder: mrc.core.segment.Builder, name: str) -> None: ...
    def get_source(self, arg0: object) -> ControlMessageWritableAcceptor: ...
    pass
class ControlMessageWritableAcceptor(mrc.core.common.WritableAcceptor):
    pass
class ControlMessageWritableProvider(mrc.core.common.WritableProvider):
    pass
