from __future__ import annotations

from .core import CoreMixin
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
from .helpers import HelpersMixin
from .imports import ImportsMixin
from .expressions import ExpressionsMixin
from .oop import OopMixin


class CCodeGenerator(
    CoreMixin,
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
    HelpersMixin,
    ImportsMixin,
    ExpressionsMixin,
    OopMixin,
):
    """Public C backend façade.

    The implementation is split by responsibility into mixins.  Existing callers
    can continue to instantiate CCodeGenerator and call generate_from_json().
    """

    pass
