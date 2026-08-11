"""Compatibility façade for the Phils Ocean C backend v0.2.

Existing callers may keep using::

    from src.compiler import CCodeGenerator

The implementation lives in ``src.codegen``.
"""

from src.codegen import CCodeGenerator, OwnershipError
from src.typed_ir import IRType, TypedModule, build_typed_ir

__all__ = ["CCodeGenerator", "OwnershipError", "IRType", "TypedModule", "build_typed_ir"]
