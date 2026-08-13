from __future__ import annotations

from .core import CoreMixin
from .naming import NamingMixin
from .ownership import OwnershipMixin
from .scope import ScopeMixin
from .types import TypesMixin
from .orchestrator import OrchestratorMixin
from .statements import StatementsMixin
from .calls import CallsMixin
from .indexing import IndexingMixin
from .io import IoMixin
from .list_codegen import ListCodegenMixin
from .tuple_codegen import TupleCodegenMixin
from .dict_codegen import DictCodegenMixin
from .array_codegen import ArrayCodegenMixin
from .tensor_codegen import TensorCodegenMixin
from .helpers import HelpersMixin
from .imports import ImportsMixin
from .expressions import ExpressionsMixin
from .oop import OopMixin


class CCodeGenerator(
    CoreMixin,
    NamingMixin,
    OwnershipMixin,
    ScopeMixin,
    TypesMixin,
    OrchestratorMixin,
    StatementsMixin,
    CallsMixin,
    IndexingMixin,
    IoMixin,
    ListCodegenMixin,
    TupleCodegenMixin,
    DictCodegenMixin,
    ArrayCodegenMixin,
    TensorCodegenMixin,
    HelpersMixin,
    ImportsMixin,
    ExpressionsMixin,
    OopMixin,
):
    """Public C backend façade.

    The implementation is split by responsibility into mixins. New compiler
    passes should call ``generate_from_typed_ir``; existing callers can keep
    using ``generate_from_json`` through its compatibility adapter.
    """

    pass
