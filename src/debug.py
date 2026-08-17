import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Dict, List, Optional

from src.modules.constants import DATA_TYPES, KNOWN_C_TYPES
from src.parsing.type_system import TENSOR_DTYPES
from src.modules.logger import logger
from src.diagnostics import Diagnostic, DiagnosticReport, DiagnosticSeverity, SourceLocation
from src.typed_ir import TypedModule


class Validator:
    """Validate the parser's typed graph before C code generation.

    The validator is intentionally split into small phases: symbol collection,
    structural/type checks, ownership data-flow, and diagnostics.  The parser
    emits a transitional AST, so every phase must tolerate missing optional
    fields and must never rely on a second, implicit parser pass.
    """

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.diagnostics = []
        self.scope_symbols = {}  # {scope_level: {var_name: var_info}}
        self.all_scopes = []  # Сохраняем все scopes для поиска родительских
        self.scopes_info = []  # Compatibility alias used by older helpers
        self._active_scope = None
        self.functions = {}  # {func_name: func_info}
        self.external_c_functions = set()
        self.source_map = {}  # Сопоставление узлов с исходными строками
        self.builtin_functions = {
            "print": {"return_type": "None", "min_args": 0, "max_args": None},
            "len": {"return_type": "int", "min_args": 1, "max_args": 1},
            "str": {"return_type": "str", "min_args": 1, "max_args": 1},
            "int": {"return_type": "int", "min_args": 1, "max_args": 1},
            "bool": {"return_type": "bool", "min_args": 1, "max_args": 1},
            "range": {"return_type": "range", "min_args": 1, "max_args": 3},
        }
        # Для отслеживания состояния переменных
        self.variable_history = {}  # {(scope_level, var_name): [{"action": "declare"/"assign"/"delete", "node_id": str}]}
        self.variable_states = {}  # {(scope_level, var_name): "active"/"deleted"}
        self.classes = {}  # {class_name: class_info}
        self.typed_ir = None

    def validate(self, typed_ir: TypedModule) -> list[Dict]:
        """Validate the canonical semantic module before C lowering."""
        if not isinstance(typed_ir, TypedModule):
            raise TypeError("validate expects a TypedModule")
        self.typed_ir = typed_ir
        return self._validate_scopes(list(typed_ir.backend_scopes()))

    def _validate_scopes(self, scopes: List[Mapping]) -> list[Dict]:
        """Run the existing validation passes over a typed lowering view."""
        if not isinstance(scopes, list):
            self.errors = []
            self.warnings = []
            self.diagnostics = []
            self.add_error("TypedModule scopes должны быть списком")
            return self.get_report()
        self.errors = []
        self.warnings = []
        self.diagnostics = []
        self.scope_symbols = {}
        self.functions = {}
        self.external_c_functions = set()
        self.all_scopes = scopes
        self.scopes_info = self.all_scopes
        self._active_scope = None
        self.source_map = {}
        self.variable_history = {}  # Оставляем, но не используем
        self.variable_states = {}  # Оставляем, но не используем

        if not isinstance(scopes, list):
            self.add_error("TypedModule scopes должны быть списком")
            return self.get_report()

        # Собираем информацию о всех узлах и их строках only after the input
        # shape has been checked.
        self.build_source_map(scopes)

        # Собираем информацию о всех scope'ах и символах
        self.collect_symbols(scopes)

        # ЗАКОММЕНТИРОВАТЬ: не строим историю переменных
        # self.build_variable_history(scopes)

        # Проверяем каждый scope
        for scope_idx, scope in enumerate(scopes):
            if not isinstance(scope, Mapping):
                self.add_error(f"Scope {scope_idx} должен быть объектом")
                continue
            self.validate_scope(scope, scope_idx, scopes)

        return self.get_report()

    def collect_symbols(self, scopes: List[Mapping]):
        """Собирает информацию о всех символах в системе"""
        for scope_idx, scope in enumerate(scopes):
            if not isinstance(scope, Mapping):
                continue
            for node in scope.get("graph", []):
                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/tensor/tensor_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_tensor_zeros",
                            "ocean_tensor_zeros_nd",
                            "ocean_tensor_from_cpu_strided",
                            "ocean_tensor_load_npy",
                            "ocean_tensor_save_npy",
                            "ocean_tensor_get_flat",
                            "ocean_tensor_len",
                            "ocean_tensor_copy",
                            "ocean_tensor_to",
                            "ocean_tensor_matmul",
                            "ocean_tensor_binary",
                            "ocean_tensor_scalar",
                            "ocean_tensor_reshape",
                            "ocean_tensor_reshape_2d",
                            "ocean_tensor_transpose",
                            "ocean_tensor_row",
                            "ocean_tensor_column",
                            "ocean_tensor_slice",
                            "ocean_tensor_sum",
                            "ocean_tensor_mean",
                            "ocean_tensor_max",
                            "ocean_tensor_min",
                            "ocean_tensor_item",
                            "ocean_tensor_dtype_name",
                            "ocean_tensor_is_contiguous",
                            "ocean_tensor_contiguous",
                            "ocean_tensor_fill",
                            "ocean_tensor_get_nd",
                            "ocean_tensor_set_nd",
                            "ocean_tensor_get_2d",
                            "ocean_tensor_set_2d",
                            "ocean_tensor_shape",
                            "ocean_tensor_ndim",
                            "ocean_tensor_size",
                            "ocean_tensor_device",
                            "ocean_tensor_release",
                        }
                    )
                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/tensor/autograd_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_autograd_set_requires_grad",
                            "ocean_autograd_requires_grad",
                            "ocean_autograd_has_grad",
                            "ocean_autograd_grad_copy",
                            "ocean_autograd_zero_grad",
                            "ocean_autograd_backward",
                            "ocean_autograd_binary",
                            "ocean_autograd_scalar",
                            "ocean_autograd_matmul",
                            "ocean_autograd_transpose",
                            "ocean_autograd_relu",
                            "ocean_autograd_mse_loss",
                            "ocean_autograd_parameter_uniform",
                            "ocean_autograd_sgd_step",
                        }
                    )

                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/io/file_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_file_open",
                            "ocean_file_close",
                            "ocean_file_read",
                            "ocean_file_readline",
                            "ocean_file_write",
                            "ocean_file_flush",
                            "ocean_file_eof",
                            "ocean_file_read_byte",
                            "ocean_file_write_byte",
                        }
                    )


                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/json/json_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_json_parse",
                            "ocean_json_stringify",
                            "ocean_json_release",
                            "ocean_json_new_null",
                            "ocean_json_new_bool",
                            "ocean_json_new_int",
                            "ocean_json_new_number",
                            "ocean_json_new_string",
                            "ocean_json_new_array",
                            "ocean_json_new_object",
                            "ocean_json_kind",
                            "ocean_json_is_null",
                            "ocean_json_is_bool",
                            "ocean_json_is_number",
                            "ocean_json_is_string",
                            "ocean_json_is_array",
                            "ocean_json_is_object",
                            "ocean_json_size",
                            "ocean_json_as_bool",
                            "ocean_json_as_int",
                            "ocean_json_as_float",
                            "ocean_json_as_string_copy",
                            "ocean_json_object_has",
                            "ocean_json_object_get",
                            "ocean_json_object_set",
                            "ocean_json_object_remove",
                            "ocean_json_object_key_at",
                            "ocean_json_object_value_at",
                            "ocean_json_array_get",
                            "ocean_json_array_set",
                            "ocean_json_array_append",
                        }
                    )

                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/time/time_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_time_now",
                            "ocean_time_now_ns",
                            "ocean_time_unix",
                            "ocean_time_monotonic",
                            "ocean_time_monotonic_ns",
                            "ocean_time_process",
                            "ocean_time_process_ns",
                            "ocean_time_sleep",
                            "ocean_time_sleep_ms",
                            "ocean_time_sleep_us",
                            "ocean_time_format_local",
                            "ocean_time_format_utc",
                        }
                    )

                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "math.h"
                ):
                    self.external_c_functions.update(
                        {
                            "acos",
                            "acosh",
                            "asin",
                            "asinh",
                            "atan",
                            "atan2",
                            "atanh",
                            "cbrt",
                            "ceil",
                            "copysign",
                            "cos",
                            "cosh",
                            "erf",
                            "erfc",
                            "exp",
                            "exp2",
                            "expm1",
                            "fabs",
                            "floor",
                            "fmax",
                            "fmin",
                            "fmod",
                            "hypot",
                            "isfinite",
                            "isinf",
                            "isnan",
                            "ldexp",
                            "lgamma",
                            "log",
                            "log10",
                            "log1p",
                            "log2",
                            "nan",
                            "nextafter",
                            "pow",
                            "remainder",
                            "round",
                            "scalbn",
                            "sin",
                            "sinh",
                            "sqrt",
                            "tan",
                            "tanh",
                            "tgamma",
                            "trunc",
                        }
                    )

                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/os/os_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_os_getcwd",
                            "ocean_os_chdir",
                            "ocean_os_mkdir",
                            "ocean_os_makedirs",
                            "ocean_os_remove",
                            "ocean_os_rmdir",
                            "ocean_os_rename",
                            "ocean_os_exists",
                            "ocean_os_is_file",
                            "ocean_os_is_dir",
                            "ocean_os_is_symlink",
                            "ocean_os_has_env",
                            "ocean_os_getenv_copy",
                            "ocean_os_setenv",
                            "ocean_os_unsetenv",
                            "ocean_os_pid",
                            "ocean_os_ppid",
                            "ocean_os_cpu_count",
                            "ocean_os_hostname",
                            "ocean_os_platform",
                            "ocean_os_listdir",
                            "ocean_os_dir_list_size",
                            "ocean_os_dir_list_get_copy",
                            "ocean_os_dir_list_release",
                        }
                    )

                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/logging/logging_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_logging_set_level",
                            "ocean_logging_get_level",
                            "ocean_logging_enabled",
                            "ocean_logging_set_timestamps",
                            "ocean_logging_get_timestamps",
                            "ocean_logging_to_stderr",
                            "ocean_logging_to_stdout",
                            "ocean_logging_to_file",
                            "ocean_logging_write",
                            "ocean_logging_flush",
                            "ocean_logging_shutdown",
                            "ocean_logging_set_colors",
                            "ocean_logging_get_colors",
}
                    )

                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/multiprocessing/thread_backend.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_thread_create",
                            "ocean_thread_join",
                            "ocean_thread_detach",
                            "ocean_thread_is_joinable",
                            "ocean_thread_release",
                        }
                    )
                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/net/net_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_socket_create",
                            "ocean_socket_connect",
                            "ocean_socket_bind",
                            "ocean_socket_listen",
                            "ocean_socket_accept",
                            "ocean_socket_send",
                            "ocean_socket_recv",
                            "ocean_socket_set_timeout",
                            "ocean_socket_is_open",
                            "ocean_socket_peer_address",
                            "ocean_socket_local_address",
                            "ocean_socket_close",
                            "ocean_socket_release",
                            "ocean_http_request",
                            "ocean_http_status",
                            "ocean_http_ok",
                            "ocean_http_status_text_copy",
                            "ocean_http_headers_copy",
                            "ocean_http_body_copy",
                            "ocean_http_response_release",
                        }
                    )

                if (
                    isinstance(node, Mapping)
                    and node.get("node") == "c_import"
                    and node.get("header") == "std/web/web_runtime.h"
                ):
                    self.external_c_functions.update(
                        {
                            "ocean_web_router_create",
                            "ocean_web_router_release",
                            "ocean_web_router_route",
                            "ocean_web_router_get",
                            "ocean_web_router_post",
                            "ocean_web_router_put",
                            "ocean_web_router_patch",
                            "ocean_web_router_delete",
                            "ocean_web_router_options",
                            "ocean_web_router_head",
                            "ocean_web_router_any",
                            "ocean_web_include_router",
                            "ocean_web_app_create",
                            "ocean_web_app_release",
                            "ocean_web_route",
                            "ocean_web_get",
                            "ocean_web_post",
                            "ocean_web_put",
                            "ocean_web_patch",
                            "ocean_web_delete",
                            "ocean_web_options",
                            "ocean_web_head",
                            "ocean_web_any",
                            "ocean_web_set_server_header",
                            "ocean_web_set_max_body_bytes",
                            "ocean_web_serve",
                            "ocean_web_request_method_copy",
                            "ocean_web_request_path_copy",
                            "ocean_web_request_query_copy",
                            "ocean_web_request_body_copy",
                            "ocean_web_request_remote_copy",
                            "ocean_web_request_header_copy",
                            "ocean_web_request_query_param_copy",
                            "ocean_web_request_path_param_copy",
                            "ocean_web_response_text",
                            "ocean_web_response_json",
                            "ocean_web_response_html",
                            "ocean_web_response_empty",
                            "ocean_web_response_redirect",
                            "ocean_web_response_add_header",
                            "ocean_web_response_release",
                        }
                    )

            level = scope.get("level", 0)

            if isinstance(scope.get("symbol_table"), Mapping) and scope["symbol_table"]:
                if level not in self.scope_symbols:
                    self.scope_symbols[level] = {}

                for symbol_name, symbol_info in scope["symbol_table"].items():
                    if not isinstance(symbol_info, Mapping):
                        continue
                    key = symbol_info.get("key")

                    # Сохраняем функции отдельно
                    if key == "function":
                        self.functions[symbol_name] = symbol_info
                    # Сохраняем классы отдельно
                    elif key == "class":
                        self.classes[symbol_name] = symbol_info
                        # Также добавляем в обычные символы
                        self.scope_symbols[level][symbol_name] = symbol_info
                    else:
                        # Сохраняем обычные переменные
                        self.scope_symbols[level][symbol_name] = symbol_info

            # Также собираем классы из class_declaration узлов
            if scope.get("type") == "class_declaration":
                class_name = scope.get("class_name")
                if class_name:
                    self.classes[class_name] = {
                        "name": class_name,
                        "key": "class",
                        "type": "class",
                        "value": None,
                        "id": class_name,
                        "is_deleted": False,
                    }
                    if level not in self.scope_symbols:
                        self.scope_symbols[level] = {}
                    self.scope_symbols[level][class_name] = self.classes[class_name]

    def build_source_map(self, scopes: List[Mapping]):
        """Строит карту соответствия узлов исходным строкам"""
        # Счетчик глобальных строк
        global_line_counter = 1

        for scope_idx, scope in enumerate(scopes):
            if not isinstance(scope, Mapping):
                continue
            level = scope.get("level", 0)
            graph = scope.get("graph", [])
            if not isinstance(graph, (list, tuple)):
                continue

            for node_idx, node in enumerate(graph):
                if not isinstance(node, Mapping):
                    continue
                node_id = f"{scope_idx}.{node_idx}"
                content = node.get("content", "")

                # Сохраняем информацию о строке
                self.source_map[node_id] = {
                    "content": content,
                    "scope_idx": scope_idx,
                    "scope_level": level,
                    "scope_type": scope.get("type", "unknown"),
                    "node_idx": node_idx,
                    "global_line_number": node.get("source_line") or global_line_counter,
                    "source_file": node.get("source_file"),
                    "source_column": node.get("source_column"),
                }

                # Увеличиваем счетчик только если строка не пустая
                if content.strip() and not node.get("source_line"):
                    global_line_counter += 1

    def build_variable_history(self, scopes: List[Dict]):
        """Строит историю операций с переменными С УЧЕТОМ ПОРЯДКА СТРОК"""
        # Сначала собираем все узлы в правильном порядке
        all_nodes = []

        for scope_idx, scope in enumerate(scopes):
            level = scope.get("level", 0)
            graph = scope.get("graph", [])

            for node_idx, node in enumerate(graph):
                node_id = f"{scope_idx}.{node_idx}"
                all_nodes.append(
                    {
                        "scope_idx": scope_idx,
                        "node_idx": node_idx,
                        "node_id": node_id,
                        "level": level,
                        "node": node,
                        "scope": scope,
                    }
                )

        # Сортируем по scope_idx и node_idx для сохранения порядка
        all_nodes.sort(key=lambda x: (x["scope_idx"], x["node_idx"]))

        # Глобальный счетчик времени для всех операций
        global_timestamp = 0

        for node_info in all_nodes:
            scope_idx = node_info["scope_idx"]
            node_idx = node_info["node_idx"]
            node_id = node_info["node_id"]
            level = node_info["level"]
            node = node_info["node"]
            node_type = node.get("node", "unknown")

            if node_type == "declaration":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    key = (level, symbol)
                    if key not in self.variable_history:
                        self.variable_history[key] = []
                    self.variable_history[key].append(
                        {
                            "action": "declare",
                            "node_id": node_id,
                            "content": node.get("content", ""),
                            "timestamp": global_timestamp,
                            "unique_id": f"{node_id}_{symbol}",
                        }
                    )
                    global_timestamp += 1
                    # При объявлении переменная активна
                    self.variable_states[key] = "active"

            elif node_type == "assignment":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    key = (level, symbol)
                    if key not in self.variable_history:
                        self.variable_history[key] = []
                    self.variable_history[key].append(
                        {
                            "action": "assign",
                            "node_id": node_id,
                            "content": node.get("content", ""),
                            "timestamp": global_timestamp,
                            "unique_id": f"{node_id}_{symbol}",
                        }
                    )
                    global_timestamp += 1

            elif node_type == "delete":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    key = (level, symbol)
                    if key not in self.variable_history:
                        self.variable_history[key] = []
                    self.variable_history[key].append(
                        {
                            "action": "delete",
                            "node_id": node_id,
                            "content": node.get("content", ""),
                            "timestamp": global_timestamp,
                            "unique_id": f"{node_id}_{symbol}",
                        }
                    )
                    global_timestamp += 1
                    # Переменная помечается как удаленная
                    self.variable_states[key] = "deleted"

            elif node_type == "builtin_function_call":
                func_name = node.get("function", "")
                args = node.get("arguments", [])
                dependencies = node.get("dependencies", [])

                # Используем dependencies из узла
                for dep in dependencies:
                    if dep and dep.isalpha() and dep not in ["True", "False", "None"]:
                        key = (level, dep)
                        if key not in self.variable_history:
                            self.variable_history[key] = []
                        self.variable_history[key].append(
                            {
                                "action": "use_in_function",
                                "function": func_name,
                                "node_id": node_id,
                                "content": node.get("content", ""),
                                "timestamp": global_timestamp,
                                "unique_id": f"{node_id}_{dep}",
                            }
                        )
                        global_timestamp += 1

            elif node_type == "function_call":
                args = node.get("arguments", [])
                dependencies = node.get("dependencies", [])
                for dep in dependencies:
                    if dep and dep.isalpha() and dep not in ["True", "False", "None"]:
                        key = (level, dep)
                        if key not in self.variable_history:
                            self.variable_history[key] = []
                        self.variable_history[key].append(
                            {
                                "action": "use_in_function",
                                "node_id": node_id,
                                "content": node.get("content", ""),
                                "timestamp": global_timestamp,
                                "unique_id": f"{node_id}_{dep}",
                            }
                        )
                        global_timestamp += 1

    def get_variable_state(self, var_name: str, level: int) -> str:
        """Получает текущее состояние переменной"""
        key = (level, var_name)
        return self.variable_states.get(key, "unknown")

    def is_variable_deleted(self, var_name: str, level: int) -> bool:
        """Проверяет, удалена ли переменная"""
        state = self.get_variable_state(var_name, level)
        return state == "deleted"

    def get_last_variable_action(self, var_name: str, level: int) -> Optional[Dict]:
        """Получает последнее действие с переменной"""
        key = (level, var_name)
        if key in self.variable_history and self.variable_history[key]:
            return self.variable_history[key][-1]
        return None

    def add_error(
        self,
        message: str,
        scope_idx: int = None,
        node_idx: int = None,
        source_line: int = None,
        source_content: str = None,
        code: str | None = None,
    ):
        """Добавляет ошибку с информацией о строке"""
        full_message = message
        location = self.source_map.get(f"{scope_idx}.{node_idx}") if scope_idx is not None and node_idx is not None else None
        if location:
            source_line = source_line or location.get("global_line_number")
            source_content = source_content or location.get("content")

        if source_line is not None:
            if source_content:
                full_message = f"Строка {source_line} '{source_content}': {message}"
            else:
                full_message = f"Строка {source_line}: {message}"
        elif scope_idx is not None and node_idx is not None:
            node_id = f"{scope_idx}.{node_idx}"
            if node_id in self.source_map:
                content = self.source_map[node_id]["content"]
                if content:
                    full_message = f"Строка '{content}': {message}"

        diagnostic = Diagnostic(
            severity=DiagnosticSeverity.ERROR,
            message=full_message,
            code=code or self._diagnostic_code(message, DiagnosticSeverity.ERROR),
            location=SourceLocation(
                source_file=location.get("source_file") if location else None,
                line=source_line or self.get_line_number(scope_idx, node_idx),
                column=location.get("source_column") if location else None,
                source_content=source_content or (location.get("content") if location else None),
            ),
            scope_idx=scope_idx,
            node_idx=node_idx,
        )
        self.errors.append(diagnostic)
        self.diagnostics.append(diagnostic)

    def add_warning(
        self,
        message: str,
        scope_idx: int = None,
        node_idx: int = None,
        source_line: int = None,
        source_content: str = None,
        code: str | None = None,
    ):
        """Добавляет предупреждение с информацией о строке"""
        full_message = message
        location = self.source_map.get(f"{scope_idx}.{node_idx}") if scope_idx is not None and node_idx is not None else None

        if scope_idx is not None and node_idx is not None:
            node_id = f"{scope_idx}.{node_idx}"
            if node_id in self.source_map:
                content = self.source_map[node_id]["content"]
                if content:
                    full_message = f"Строка '{content}': {message}"

        diagnostic = Diagnostic(
            severity=DiagnosticSeverity.WARNING,
            message=full_message,
            code=code or self._diagnostic_code(message, DiagnosticSeverity.WARNING),
            location=SourceLocation(
                source_file=location.get("source_file") if location else None,
                line=source_line or self.get_line_number(scope_idx, node_idx),
                column=location.get("source_column") if location else None,
                source_content=source_content or (location.get("content") if location else None),
            ),
            scope_idx=scope_idx,
            node_idx=node_idx,
        )
        self.warnings.append(diagnostic)
        self.diagnostics.append(diagnostic)

    @staticmethod
    def _diagnostic_code(message: str, severity: DiagnosticSeverity) -> str:
        """Assign a stable category to legacy validation call sites."""
        text = message.lower()
        prefix = "E" if severity is DiagnosticSeverity.ERROR else "W"
        categories = (
            ("unsafe", "100"),
            ("borrow", "200"),
            ("ссыл", "200"),
            ("moved", "201"),
            ("перемещ", "201"),
            ("deleted", "202"),
            ("удален", "202"),
            ("openmp", "300"),
            ("parallel", "300"),
            ("тип", "400"),
            ("type", "400"),
            ("не объяв", "500"),
            ("undeclared", "500"),
        )
        for marker, suffix in categories:
            if marker in text:
                return f"{prefix}{suffix}"
        return f"{prefix}000"

    def get_line_number(self, scope_idx: int, node_idx: int) -> Optional[int]:
        """Получает номер строки исходного кода для узла"""
        if scope_idx is None or node_idx is None:
            return None

        node_id = f"{scope_idx}.{node_idx}"
        if node_id in self.source_map:
            content = self.source_map[node_id]["content"]
            if content:
                return self.source_map[node_id].get("global_line_number") or node_idx + 1

        return None

    def validate_scope(self, scope: Dict, scope_idx: int, all_scopes: List[Dict]):
        """Валидирует отдельный scope с учетом всех новых проверок"""
        self._active_scope = scope
        level = scope.get("level", 0)
        scope_type = scope.get("type", "unknown")

        # 1. БАЗОВЫЕ ПРОВЕРКИ СТРУКТУРЫ
        required_fields = ["level", "type", "local_variables", "graph", "symbol_table"]
        for field in required_fields:
            if field not in scope:
                self.add_error(
                    f"Scope {scope_idx} (level {level}, type {scope_type}) отсутствует поле '{field}'",
                    scope_idx,
                    None,
                )

        # 2. ПРОВЕРКА ТАБЛИЦЫ СИМВОЛОВ И ЛОКАЛЬНЫХ ПЕРЕМЕННЫХ
        self.validate_symbol_table(scope, scope_idx)
        self.check_duplicate_declarations_in_scope(scope, scope_idx)

        # 3. КОНТЕКСТНЫЕ ПРОВЕРКИ В ЗАВИСИМОСТИ ОТ ТИПА SCOPE
        if scope_type == "class_declaration":
            # Ключевая проверка наследования для классов
            self.validate_inheritance_hierarchy(scope, scope_idx)

            # Вычисляем MRO для информации и дальнейших проверок
            class_name = scope.get("class_name")
            if class_name:
                mro = self.check_method_resolution_order(class_name)
                # Можно сохранить MRO для использования в других проверках
                if mro and len(mro) > 1:
                    logger.debug(f"  MRO для класса '{class_name}': {mro}")

        elif scope_type == "function":
            # Проверяем функции для потоков
            self.validate_thread_functions(scope, scope_idx)

            # Старая проверка наличия return
            self.validate_function_return(scope, scope_idx)
            # Проверка типа возвращаемого значения
            self.validate_function_return_type(scope, scope_idx)
            # Проверка всех путей возврата
            self.validate_return_paths(scope, scope_idx)
            # Ownership/borrow data-flow pass.  The legacy graph checks above
            # remain for compatibility; this pass tracks lifetimes explicitly.
            self.validate_lifetimes(scope, scope_idx)

        # Проверка неиспользуемых параметров для всех типов функций/методов
        if scope_type in ["function", "constructor", "class_method"]:
            self.check_unused_parameters(scope, scope_idx)

        # 4. ПРОВЕРКА ГРАФА ОПЕРАЦИЙ
        if "graph" in scope:
            self.validate_graph(scope, scope_idx)

            # Детальная проверка каждого узла графа
            graph = scope.get("graph", [])
            for node_idx, node in enumerate(graph):
                node_type = node.get("node", "unknown")

                # ВАЖНЕЙШИЕ ПРОВЕРКИ УЗЛОВ:

                # 4.1 ПРОВЕРКА УКАЗАТЕЛЕЙ (критически важно для C-кода)
                self.validate_pointer_usage(node, node_idx, scope_idx, level)

                # 4.2 ПРОВЕРКА ГРАНИЦ МАССИВОВ (предотвращение ошибок выполнения)
                self.validate_array_bounds(node, node_idx, scope_idx, level)

                # 4.3 ПРОВЕРКА СТРОКОВЫХ ОПЕРАЦИЙ (частая ошибка)
                self.validate_string_operations(node, node_idx, scope_idx, level)

                # 4.4 ПРОВЕРКА C-ФУНКЦИЙ (безопасность и корректность)
                self.validate_unsafe_boundary(node, node_idx, scope_idx)
                self.validate_c_function_calls(node, node_idx, scope_idx, level)

                # 4.5 Дополнительная проверка деления на ноль.  Structural
                # and expression type checks already run once in validate_graph.
                self.check_division_by_zero(node, node_idx, scope_idx, level)

                # 4.7 ПРОВЕРКА ОПЕРАЦИЙ С КОРТЕЖАМИ (неизменяемость)
                if node_type in [
                    "index_assignment",
                    "augmented_index_assignment",
                    "slice_assignment",
                ]:
                    variable = node.get("variable", "")
                    if variable:
                        var_info = self.get_symbol_info(variable, level)
                        if var_info and "tuple" in var_info.get("type", ""):
                            self.add_error(
                                f"попытка изменения неизменяемого кортежа '{variable}'",
                                scope_idx,
                                node_idx,
                            )

            # Method lookup is a scope-level check; running it once avoids
            # repeating the same warning for every unrelated graph node.
            self.check_undefined_methods(scope, scope_idx)

            # 5. ПОСТ-ПРОВЕРКИ ПОСЛЕ АНАЛИЗА ГРАФА

            # 5.1 Проверка неиспользуемых переменных
            self.check_unused_variables(scope, scope_idx)

            # 5.2 Проверка условий циклов
            self.check_loop_conditions(scope, scope_idx)

            # 5.3 Проверка утечек памяти (особенно для указателей)
            self.check_memory_leaks(scope, scope_idx)

            # 5.4 Проверка отсутствующих объявлений (C-типы, импорты)
            self.check_missing_declarations(scope, scope_idx)

        # 6. СПЕЦИФИЧНЫЕ ПРОВЕРКИ ДЛЯ ЦИКЛОВ
        self.validate_loops(scope, scope_idx)

        # 7. ПРОВЕРКА ВЫЗОВОВ МЕТОДОВ КЛАССОВ (особенно в наследовании)
        if scope_type == "class_method":
            self._validate_class_method_calls(scope, scope_idx)

        # 8. СБОР МЕТРИК КОДА (информация для разработчика)
        if scope_type == "function" or scope_type == "class_method":
            self._collect_function_metrics(scope, scope_idx)

    def validate_unsafe_boundary(self, node: Dict, node_idx: int, scope_idx: int):
        """Reject raw pointers and direct C FFI unless explicitly marked unsafe."""
        node_type = node.get("node")
        if node_type == "c_call" and not node.get("unsafe", False):
            self.add_error(
                "direct C calls require an explicit unsafe: block",
                scope_idx,
                node_idx,
            )

        var_type = node.get("var_type", "")
        type_info = node.get("type_info", {}) or {}
        if (
            node_type in {"declaration", "redeclaration"}
            and (type_info.get("kind") == "raw_pointer" or var_type.startswith("*"))
            and not node.get("unsafe", False)
        ):
            self.add_error(
                "raw pointer declarations require an explicit unsafe: block",
                scope_idx,
                node_idx,
            )

        if node_type != "c_call" and self._contains_unsafe_ffi(node) and not node.get("unsafe", False):
            self.add_error(
                "C FFI expressions require an explicit unsafe: block",
                scope_idx,
                node_idx,
            )

    def _contains_unsafe_ffi(self, value) -> bool:
        if isinstance(value, str):
            return bool(re.search(r"@[A-Za-z_][A-Za-z0-9_]*\s*\(", value))
        if isinstance(value, Mapping):
            return any(self._contains_unsafe_ffi(child) for child in value.values())
        if isinstance(value, list):
            return any(self._contains_unsafe_ffi(child) for child in value)
        return False

    # ------------------------------------------------------------------
    # Ownership / lifetime data-flow
    # ------------------------------------------------------------------

    def _lifetime_state(self, py_type: str) -> Dict:
        return {
            "type": py_type or "",
            "status": "active",
            "borrow_source": None,
            "borrow_mutable": False,
            "shared_borrows": set(),
            "mutable_borrow": None,
        }

    def _lifetime_is_dead(self, state: Optional[Dict]) -> bool:
        return bool(state and state.get("status") in {"dead", "moved", "maybe_dead"})

    def _lifetime_is_unique(self, py_type: str) -> bool:
        value = (py_type or "").strip()
        if value.startswith("&mut "):
            value = value[5:].strip()
        elif value.startswith("&"):
            value = value[1:].strip()
        return value.startswith("array[")

    def _lifetime_variable_names(self, value) -> set[str]:
        """Collect variable references from expression ASTs only."""
        names: set[str] = set()
        if isinstance(value, list):
            for item in value:
                names.update(self._lifetime_variable_names(item))
            return names
        if not isinstance(value, Mapping):
            return names

        ast_type = value.get("type")
        if ast_type == "variable":
            name = value.get("name") or value.get("value")
            if isinstance(name, str):
                names.add(name)
            return names
        if ast_type in {"literal", "empty", "unknown"}:
            return names

        if ast_type in {"attribute_access", "complex_attribute_access", "method_call"}:
            obj = value.get("object")
            if isinstance(obj, str):
                names.add(obj)

        for key, child in value.items():
            if key in {
                "type", "name", "value", "attribute", "function", "method",
                "operator", "operator_symbol", "data_type", "original",
            }:
                continue
            names.update(self._lifetime_variable_names(child))
        return names

    def _lifetime_node_reads(self, node: Dict) -> set[str]:
        names: set[str] = set()
        for dependency in node.get("dependencies", []) or []:
            if isinstance(dependency, str) and re.match(r"^[A-Za-z_]\w*$", dependency):
                names.add(dependency)

        for key in (
            "expression_ast", "condition", "condition_ast", "iterable",
            "arguments", "argument", "value", "index", "indices",
        ):
            names.update(self._lifetime_variable_names(node.get(key)))

        obj = node.get("object")
        if isinstance(obj, str) and re.match(r"^[A-Za-z_]\w*$", obj):
            names.add(obj)
        return names

    def _lifetime_error(self, message: str, scope_idx: int, node_idx: int) -> None:
        self.add_error(
            message,
            scope_idx,
            node_idx,
            source_line=getattr(self, "_lifetime_source_line", None),
            source_content=getattr(self, "_lifetime_source_content", None),
        )

    def _lifetime_check_reads(
        self, names: set[str], env: Dict[str, Dict], scope_idx: int, node_idx: int
    ) -> None:
        for name in sorted(names):
            state = env.get(name)
            if not state:
                continue
            if state.get("status") == "dead":
                self._lifetime_error(
                    f"use of deleted value '{name}'", scope_idx, node_idx
                )
            elif state.get("status") == "moved":
                self._lifetime_error(
                    f"use of moved value '{name}'", scope_idx, node_idx
                )
            elif state.get("status") == "maybe_dead":
                self._lifetime_error(
                    f"value '{name}' may be deleted or moved on another control-flow path",
                    scope_idx,
                    node_idx,
                )
            elif state.get("mutable_borrow") and state.get("borrow_source") is None:
                self._lifetime_error(
                    f"cannot read owner '{name}' while a mutable borrow is active",
                    scope_idx,
                    node_idx,
                )

    def _lifetime_check_mutation(
        self, name: str, env: Dict[str, Dict], scope_idx: int, node_idx: int
    ) -> None:
        state = env.get(name)
        if not state:
            return
        if self._lifetime_is_dead(state):
            self._lifetime_check_reads({name}, env, scope_idx, node_idx)
            return
        if state.get("borrow_source") and not state.get("borrow_mutable"):
            self._lifetime_error(
                f"cannot mutate through immutable borrow '{name}'", scope_idx, node_idx
            )
        elif state.get("borrow_source"):
            return
        elif state.get("shared_borrows") or state.get("mutable_borrow"):
            self._lifetime_error(
                f"cannot mutate '{name}' while it is borrowed", scope_idx, node_idx
            )

    def _lifetime_release_borrow(self, name: str, env: Dict[str, Dict]) -> None:
        state = env.get(name)
        if not state or not state.get("borrow_source"):
            return
        source = env.get(state["borrow_source"])
        if source:
            if state.get("borrow_mutable"):
                if source.get("mutable_borrow") == name:
                    source["mutable_borrow"] = None
            else:
                source.setdefault("shared_borrows", set()).discard(name)

    def _lifetime_register_borrow(
        self,
        target: str,
        source_name: str,
        mutable: bool,
        env: Dict[str, Dict],
        scope_idx: int,
        node_idx: int,
    ) -> None:
        source = env.get(source_name)
        if not source or self._lifetime_is_dead(source):
            self._lifetime_error(
                f"cannot borrow dead value '{source_name}'", scope_idx, node_idx
            )
            return
        if source.get("borrow_source"):
            self._lifetime_error(
                f"reborrowing '{source_name}' is not allowed", scope_idx, node_idx
            )
            return
        if mutable and (source.get("shared_borrows") or source.get("mutable_borrow")):
            self._lifetime_error(
                f"cannot mutably borrow '{source_name}': another borrow is active",
                scope_idx,
                node_idx,
            )
            return
        if not mutable and source.get("mutable_borrow"):
            self._lifetime_error(
                f"cannot immutably borrow '{source_name}': mutable borrow is active",
                scope_idx,
                node_idx,
            )
            return

        borrow = env[target]
        borrow["borrow_source"] = source_name
        borrow["borrow_mutable"] = mutable
        if mutable:
            source["mutable_borrow"] = target
        else:
            source.setdefault("shared_borrows", set()).add(target)

    def _lifetime_call_escape_check(
        self, node: Dict, env: Dict[str, Dict], scope_idx: int, node_idx: int
    ) -> None:
        function_name = node.get("function", "")
        arguments = node.get("arguments", []) or []
        parameter_types = []
        for candidate in self.all_scopes:
            if candidate.get("type") == "function" and candidate.get("function_name") == function_name:
                parameter_types = [p.get("type", "") for p in candidate.get("parameters", [])]
                break

        for index, argument in enumerate(arguments):
            argument_names = self._lifetime_variable_names(argument)
            for name in argument_names:
                state = env.get(name)
                if not state:
                    continue
                expected = parameter_types[index] if index < len(parameter_types) else ""
                if state.get("borrow_source"):
                    if node.get("node") == "c_call" or not expected.startswith("&"):
                        self._lifetime_error(
                            f"borrow '{name}' escapes through call '{function_name}'",
                            scope_idx,
                            node_idx,
                        )
                    elif expected.startswith("&mut ") and not state.get("borrow_mutable"):
                        self._lifetime_error(
                            f"immutable borrow '{name}' cannot be passed to mutable parameter",
                            scope_idx,
                            node_idx,
                        )
                    continue

                # Passing an owned array/tensor by value transfers its owner
                # into the callee.  The binding is dead after the call; the
                # caller must use ``&T``/``&mut T`` for a non-consuming call.
                if expected and not expected.startswith("&") and self._lifetime_is_unique(expected):
                    if self._lifetime_is_unique(state.get("type", "")):
                        self._lifetime_check_mutation(name, env, scope_idx, node_idx)
                        if not self._lifetime_is_dead(state):
                            state["status"] = "moved"

    def _lifetime_analyze_node(
        self, node: Dict, env: Dict[str, Dict], scope_idx: int, node_idx: int
    ) -> None:
        self._lifetime_source_line = node.get("source_line")
        self._lifetime_source_content = node.get("content")
        node_type = node.get("node", "")
        reads = self._lifetime_node_reads(node)

        if node_type == "declaration":
            target = node.get("var_name", "")
            var_type = node.get("var_type", "")
            self._lifetime_check_reads(reads, env, scope_idx, node_idx)
            env[target] = self._lifetime_state(var_type)
            borrow_effects = [
                effect
                for effect in getattr(node, "ownership_effects", ())
                if effect.kind == "borrow"
            ]
            move_effects = [
                effect
                for effect in getattr(node, "ownership_effects", ())
                if effect.kind == "move" and effect.source
            ]
            if borrow_effects:
                effect = borrow_effects[0]
                self._lifetime_register_borrow(
                    target,
                    effect.source or "",
                    effect.mutable,
                    env,
                    scope_idx,
                    node_idx,
                )
            elif move_effects:
                for effect in move_effects:
                    source = effect.source
                    if source and source in env:
                        self._lifetime_check_mutation(source, env, scope_idx, node_idx)
                        env[source]["status"] = "moved"
            elif self._lifetime_is_unique(var_type):
                source_ast = node.get("expression_ast") or {}
                source = source_ast.get("name") or source_ast.get("value") if source_ast.get("type") == "variable" else None
                if source and source != target and source in env:
                    self._lifetime_check_mutation(source, env, scope_idx, node_idx)
                    env[source]["status"] = "moved"
            if "&" in var_type and not var_type.strip().startswith("&"):
                self._lifetime_error(
                    f"borrow cannot be stored inside '{var_type}'", scope_idx, node_idx
                )
            return

        if node_type in {"assignment", "augmented_assignment"}:
            self._lifetime_check_reads(reads, env, scope_idx, node_idx)
            target = (node.get("symbols") or [""])[0]
            self._lifetime_check_mutation(target, env, scope_idx, node_idx)
            state = env.get(target)
            move_effects = [
                effect
                for effect in getattr(node, "ownership_effects", ())
                if effect.kind == "move" and effect.source
            ]
            if move_effects:
                for effect in move_effects:
                    source = effect.source
                    if source and source in env:
                        env[source]["status"] = "moved"
                if state:
                    state["status"] = "active"
            elif state and self._lifetime_is_unique(state.get("type", "")):
                ast = node.get("expression_ast") or {}
                source = (ast.get("name") or ast.get("value")) if ast.get("type") == "variable" else None
                if source and source != target and source in env:
                    env[source]["status"] = "moved"
                    state["status"] = "active"
            return

        if node_type == "delete":
            drop_effects = [
                effect
                for effect in getattr(node, "ownership_effects", ())
                if effect.kind == "drop" and effect.target
            ]
            targets = [effect.target for effect in drop_effects] or node.get("symbols", [])
            for target in targets:
                state = env.get(target)
                if not state:
                    continue
                if state.get("borrow_source"):
                    self._lifetime_release_borrow(target, env)
                    state["status"] = "dead"
                elif state.get("shared_borrows") or state.get("mutable_borrow"):
                    self._lifetime_error(
                        f"cannot delete '{target}' while it is borrowed", scope_idx, node_idx
                    )
                else:
                    state["status"] = "dead"
            return

        if node_type == "return":
            self._lifetime_check_reads(reads, env, scope_idx, node_idx)
            return_ast = (node.get("operations") or [{}])[0].get("value") or {}
            return_type = self._function_scope_return_type(scope_idx)
            if return_type.startswith("&") or any(
                env.get(name, {}).get("borrow_source") for name in self._lifetime_variable_names(return_ast)
            ):
                self._lifetime_error(
                    "borrow cannot escape through a function return", scope_idx, node_idx
                )
            move_sources = {
                effect.source
                for effect in getattr(node, "ownership_effects", ())
                if effect.kind == "move" and effect.source
            }
            for name in move_sources or self._lifetime_variable_names(return_ast):
                state = env.get(name)
                if state and self._lifetime_is_unique(state.get("type", "")) and not state.get("borrow_source"):
                    state["status"] = "moved"
            return

        if node_type in {
            "index_assignment", "nested_index_assignment", "slice_assignment",
            "attribute_assignment",
        }:
            self._lifetime_check_reads(reads, env, scope_idx, node_idx)
            target = node.get("variable") or node.get("object") or ""
            if isinstance(target, str):
                self._lifetime_check_mutation(target, env, scope_idx, node_idx)
            return

        if node_type == "method_call":
            self._lifetime_check_reads(reads, env, scope_idx, node_idx)
            if node.get("method") in {"append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse"}:
                self._lifetime_check_mutation(node.get("object", ""), env, scope_idx, node_idx)
            return

        if node_type in {"function_call", "function_call_assignment", "c_call", "builtin_function_call"}:
            self._lifetime_check_reads(reads, env, scope_idx, node_idx)
            if node_type != "builtin_function_call":
                self._lifetime_call_escape_check(node, env, scope_idx, node_idx)
                for effect in getattr(node, "ownership_effects", ()):
                    if effect.kind != "move" or not effect.source:
                        continue
                    source_state = env.get(effect.source)
                    if source_state:
                        self._lifetime_check_mutation(
                            effect.source, env, scope_idx, node_idx
                        )
                        if not self._lifetime_is_dead(source_state):
                            source_state["status"] = "moved"
            return

        self._lifetime_check_reads(reads, env, scope_idx, node_idx)

    def _function_scope_return_type(self, scope_idx: int) -> str:
        if 0 <= scope_idx < len(self.all_scopes):
            return self.all_scopes[scope_idx].get("return_type", "None")
        return "None"

    def _lifetime_cleanup_block(self, env: Dict[str, Dict], outer_names: set[str]) -> None:
        for name in list(env):
            if name not in outer_names:
                self._lifetime_release_borrow(name, env)
                del env[name]

    def _lifetime_merge(self, base: Dict[str, Dict], branches: list[Dict[str, Dict]]) -> Dict[str, Dict]:
        merged = deepcopy(base)
        for name, state in merged.items():
            statuses = [branch.get(name, state).get("status", "active") for branch in branches]
            if all(status == statuses[0] for status in statuses):
                state["status"] = statuses[0]
            elif any(status in {"dead", "moved", "maybe_dead"} for status in statuses):
                state["status"] = "maybe_dead"
            state["shared_borrows"] = set()
            state["mutable_borrow"] = None
        return merged

    def _lifetime_analyze_sequence(
        self,
        nodes: list[Dict],
        env: Dict[str, Dict],
        scope_idx: int,
        *,
        block: bool = False,
    ) -> None:
        outer_names = set(env)
        for node_idx, node in enumerate(nodes):
            self._lifetime_source_line = node.get("source_line")
            self._lifetime_source_content = node.get("content")
            node_type = node.get("node", "")
            if node_type == "if_statement":
                self._lifetime_check_reads(self._lifetime_node_reads(node), env, scope_idx, node_idx)
                branch_envs = []
                for body in [node.get("body", [])] + [item.get("body", []) for item in node.get("elif_blocks", [])]:
                    branch = deepcopy(env)
                    self._lifetime_analyze_sequence(body, branch, scope_idx, block=True)
                    branch_envs.append(branch)
                else_block = node.get("else_block")
                if else_block is not None:
                    branch = deepcopy(env)
                    self._lifetime_analyze_sequence(else_block.get("body", []), branch, scope_idx, block=True)
                    branch_envs.append(branch)
                else:
                    branch_envs.append(deepcopy(env))
                env.update(self._lifetime_merge(env, branch_envs))
            elif node_type in {"while_loop", "for_loop"}:
                self._lifetime_check_reads(self._lifetime_node_reads(node), env, scope_idx, node_idx)
                body = deepcopy(env)
                if node_type == "for_loop":
                    loop_var = node.get("loop_variable", "")
                    if loop_var:
                        body[loop_var] = self._lifetime_state("int")
                self._lifetime_analyze_sequence(node.get("body", []), body, scope_idx, block=True)
                env.update(self._lifetime_merge(env, [env, body]))
            else:
                self._lifetime_analyze_node(node, env, scope_idx, node_idx)
        if block:
            self._lifetime_cleanup_block(env, outer_names)

    def validate_lifetimes(self, scope: Dict, scope_idx: int) -> None:
        env: Dict[str, Dict] = {}
        for parameter in scope.get("parameters", []) or []:
            name = parameter.get("name", "")
            if name:
                env[name] = self._lifetime_state(parameter.get("type", ""))
        self._lifetime_analyze_sequence(scope.get("graph", []), env, scope_idx)

    def check_duplicate_declarations_in_scope(self, scope: Dict, scope_idx: int):
        """Проверяет дублирование переменных в local_variables"""
        local_vars = scope.get("local_variables", [])
        seen = {}

        for i, var_name in enumerate(local_vars):
            if var_name in seen:
                # Нашли дубликат
                first_occurrence = seen[var_name]
                self.add_warning(
                    f"переменная '{var_name}' дублируется в local_variables (первое упоминание на позиции {first_occurrence})",
                    scope_idx,
                    None,
                )
            else:
                seen[var_name] = i

    def validate_symbol_table(self, scope: Dict, scope_idx: int):
        """Валидирует таблицу символов scope'а"""
        symbol_table = scope.get("symbol_table", {})
        local_vars = scope.get("local_variables", [])

        for var_name in local_vars:
            if var_name not in symbol_table:
                self.add_warning(
                    f"переменная '{var_name}' в local_variables отсутствует в symbol_table",
                    scope_idx,
                    None,
                )

        for symbol_name, symbol_info in symbol_table.items():
            self.validate_symbol(symbol_info, scope_idx, symbol_name)

    def validate_symbol(self, symbol_info: Dict, scope_idx: int, symbol_name: str):
        """Валидирует отдельный символ"""
        required_fields = ["name", "key", "type", "id"]
        for field in required_fields:
            if field not in symbol_info:
                self.add_error(
                    f"у символа '{symbol_name}' отсутствует поле '{field}'",
                    scope_idx,
                    None,
                )

        if symbol_info.get("name") != symbol_name:
            self.add_warning(
                f"имя символа '{symbol_name}' не совпадает с полем 'name': {symbol_info.get('name')}",
                scope_idx,
                None,
            )

        var_type = symbol_info.get("type", "")
        key = symbol_info.get("key", "")

        # Для классов и функций типы могут быть любыми
        if key in ["class", "function"]:
            return

        # Пропускаем проверку для параметра 'self'
        if symbol_name == "self":
            return

        normalized_type = var_type.strip()
        while normalized_type.startswith(("&mut ", "&")):
            if normalized_type.startswith("&mut "):
                normalized_type = normalized_type[5:].strip()
            else:
                normalized_type = normalized_type[1:].strip()
        if normalized_type.startswith("tensor["):
            self.add_error(
                "тип tensor[T] удален; используйте публичный Tensor[T]",
                scope_idx,
                None,
            )

        type_info = symbol_info.get("type_info") or {}
        if var_type.startswith("Tensor[") and var_type.endswith("]"):
            tensor_dtype = var_type[len("Tensor[") : -1].strip()
            if tensor_dtype not in TENSOR_DTYPES:
                self.add_error(
                    f"Tensor dtype '{tensor_dtype}' must be a numeric scalar type",
                    scope_idx,
                    None,
                )
        known_generic = (
            type_info.get("kind") in {"generic", "borrow", "mut_borrow", "raw_pointer", "optional"}
            or var_type.startswith(("list[", "dict[", "tuple[", "array[", "shared["))
        )
        if (
            var_type not in DATA_TYPES
            and var_type not in KNOWN_C_TYPES
            and not var_type.startswith("*")
            and not known_generic
        ):
            # Проверяем, не является ли это пользовательским классом
            if var_type not in self.classes:
                self.add_warning(
                    f"символ '{symbol_name}' имеет неизвестный тип '{var_type}'",
                    scope_idx,
                    None,
                )

    def validate_graph(self, scope: Dict, scope_idx: int):
        """Валидирует граф операций"""
        graph = scope.get("graph", [])
        symbol_table = scope.get("symbol_table", {})
        level = scope.get("level", 0)

        # Отслеживаем состояние переменных в процессе валидации
        variable_states = {
            parameter.get("name"): "active"
            for parameter in scope.get("parameters", []) or []
            if parameter.get("name")
        }  # {var_name: "active"/"deleted"/"pointer_deleted"}

        for node_idx, node in enumerate(graph):
            node_type = node.get("node", "unknown")
            content = node.get("content", "")

            self.validate_node_types(node, node_idx, scope_idx, level)

            if node_type == "declaration":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    if symbol in variable_states:
                        if variable_states[symbol] == "deleted":
                            # Переменная была удалена, можно переобъявлять
                            variable_states[symbol] = "active"
                        else:
                            # Переменная уже активна - ошибка
                            self.add_error(
                                f"повторное объявление переменной '{symbol}' без предварительного удаления",
                                scope_idx,
                                node_idx,
                            )
                    else:
                        # Новая переменная
                        variable_states[symbol] = "active"

                self.validate_declaration(
                    node, node_idx, scope_idx, symbol_table, level
                )

            elif node_type == "redeclaration":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    if symbol in variable_states:
                        if variable_states[symbol] == "deleted":
                            # Разрешено: переменная была удалена
                            variable_states[symbol] = "active"
                        else:
                            # Ошибка: переменная активна
                            self.add_error(
                                f"недопустимое переобъявление переменной '{symbol}' без предварительного удаления",
                                scope_idx,
                                node_idx,
                            )
                    else:
                        # Переменная не была объявлена в этом scope
                        # Может быть объявлена в родительском scope
                        self.add_warning(
                            f"переобъявление переменной '{symbol}', не объявленной в текущем scope",
                            scope_idx,
                            node_idx,
                        )
                        variable_states[symbol] = "active"

                self.validate_declaration(
                    node, node_idx, scope_idx, symbol_table, level
                )

            elif node_type in ["assignment", "declaration", "return", "while_loop"]:
                self.validate_node_types(node, node_idx, scope_idx, level)

            elif node_type == "delete":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    if symbol in variable_states:
                        if variable_states[symbol] == "deleted":
                            # Уже удалена
                            self.add_error(
                                f"переменная '{symbol}' уже была удалена",
                                scope_idx,
                                node_idx,
                            )
                        else:
                            # Помечаем как удаленную
                            variable_states[symbol] = "deleted"
                            logger.debug(
                                f"    Переменная '{symbol}' помечена как удаленная"
                            )
                    else:
                        # Переменная не была объявлена в этом scope
                        self.add_error(
                            f"переменная '{symbol}' не была объявлена перед удалением",
                            scope_idx,
                            node_idx,
                        )

                self.validate_delete(node, node_idx, scope_idx, symbol_table, level)

            elif node_type == "builtin_function_call":
                # Проверяем аргументы
                func_name = node.get("function", "")
                dependencies = node.get("dependencies", [])

                for dep in dependencies:
                    if dep and dep.isalpha() and dep not in ["True", "False", "None"]:
                        # Проверяем в текущем scope
                        if dep in variable_states:
                            if variable_states[dep] == "deleted":
                                self.add_error(
                                    f"использование удаленной переменной '{dep}' в аргументе функции '{func_name}'",
                                    scope_idx,
                                    node_idx,
                                )
                            elif variable_states[dep] == "pointer_deleted":
                                self.add_warning(
                                    f"использование удаленного указателя '{dep}' в аргументе функции '{func_name}' (данные могут быть доступны)",
                                    scope_idx,
                                    node_idx,
                                )
                        else:
                            # Проверяем в родительских scope'ах
                            parent_scope = self.get_parent_scope(level)
                            found = False
                            current_level = level

                            while not found and parent_scope is not None:
                                parent_vars = parent_scope.get("local_variables", [])
                                if dep in parent_vars:
                                    found = True
                                    break

                                current_level = parent_scope.get("level", 0)
                                parent_scope = self.get_parent_scope(current_level)

                            if not found:
                                self.add_error(
                                    f"использование необъявленной переменной '{dep}' в аргументе функции '{func_name}'",
                                    scope_idx,
                                    node_idx,
                                )

                self.validate_builtin_function_call(
                    node, node_idx, scope_idx, symbol_table, level
                )

            elif node_type == "print":
                dependencies = node.get("dependencies", [])
                for dep in dependencies:
                    if dep and dep.isalpha() and dep not in ["True", "False", "None"]:
                        # Проверяем в текущем scope
                        if dep in variable_states:
                            if variable_states[dep] == "deleted":
                                self.add_error(
                                    f"использование удаленной переменной '{dep}' в print",
                                    scope_idx,
                                    node_idx,
                                )
                            elif variable_states[dep] == "pointer_deleted":
                                self.add_warning(
                                    f"использование удаленного указателя '{dep}' в print (данные могут быть доступны)",
                                    scope_idx,
                                    node_idx,
                                )
                        else:
                            # Проверяем в родительских scope'ах
                            parent_scope = self.get_parent_scope(level)
                            found = False
                            current_level = level

                            while not found and parent_scope is not None:
                                parent_vars = parent_scope.get("local_variables", [])
                                if dep in parent_vars:
                                    found = True
                                    break

                                current_level = parent_scope.get("level", 0)
                                parent_scope = self.get_parent_scope(current_level)

                            if not found:
                                self.add_error(
                                    f"использование необъявленной переменной '{dep}' в print",
                                    scope_idx,
                                    node_idx,
                                )

                self.validate_print(node, node_idx, scope_idx, symbol_table, level)

            elif node_type == "assignment":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    if symbol in variable_states:
                        if variable_states[symbol] == "deleted":
                            self.add_error(
                                f"присваивание полностью удаленной переменной '{symbol}' (требуется переобъявление)",
                                scope_idx,
                                node_idx,
                            )
                        elif variable_states[symbol] == "pointer_deleted":
                            self.add_warning(
                                f"присваивание удаленному указателю '{symbol}' (данные могут быть доступны)",
                                scope_idx,
                                node_idx,
                            )
                    else:
                        # Проверяем в родительских scope'ах
                        parent_scope = self.get_parent_scope(level)
                        found = False
                        current_level = level

                        while not found and parent_scope is not None:
                            parent_vars = parent_scope.get("local_variables", [])
                            if symbol in parent_vars:
                                found = True
                                break

                            current_level = parent_scope.get("level", 0)
                            parent_scope = self.get_parent_scope(current_level)

                        if not found:
                            self.add_error(
                                f"присваивание необъявленной переменной '{symbol}'",
                                scope_idx,
                                node_idx,
                            )

                self.validate_assignment(node, node_idx, scope_idx, symbol_table, level)

            elif node_type == "dereference_write":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    if symbol in variable_states:
                        if variable_states[symbol] == "deleted":
                            self.add_error(
                                f"запись через полностью удаленный указатель '{symbol}'",
                                scope_idx,
                                node_idx,
                            )
                        elif variable_states[symbol] == "pointer_deleted":
                            self.add_warning(
                                f"запись через удаленный указатель '{symbol}' (данные могут быть доступны)",
                                scope_idx,
                                node_idx,
                            )
                    else:
                        self.add_warning(
                            f"запись через необъявленный указатель '{symbol}'",
                            scope_idx,
                            node_idx,
                        )

                self.validate_dereference_write(
                    node, node_idx, scope_idx, symbol_table, level
                )

            elif node_type == "dereference_read":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    if symbol in variable_states:
                        if variable_states[symbol] == "deleted":
                            self.add_error(
                                f"чтение в полностью удаленную переменную '{symbol}'",
                                scope_idx,
                                node_idx,
                            )
                    else:
                        self.add_warning(
                            f"чтение в необъявленную переменную '{symbol}'",
                            scope_idx,
                            node_idx,
                        )

            elif node_type == "augmented_assignment":
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    if symbol in variable_states:
                        if variable_states[symbol] == "deleted":
                            self.add_error(
                                f"составное присваивание удаленной переменной '{symbol}'",
                                scope_idx,
                                node_idx,
                            )
                        elif variable_states[symbol] == "pointer_deleted":
                            self.add_warning(
                                f"составное присваивание удаленному указателю '{symbol}'",
                                scope_idx,
                                node_idx,
                            )
                    else:
                        self.add_error(
                            f"составное присваивание необъявленной переменной '{symbol}'",
                            scope_idx,
                            node_idx,
                        )

                self.validate_augmented_assignment(
                    node, node_idx, scope_idx, symbol_table, level
                )

            elif node_type in [
                "function_declaration",
                "function_call",
                "function_call_assignment",
                "return",
                "while_loop",
                "for_loop",
            ]:
                # Вызываем соответствующие методы валидации
                if node_type == "function_declaration":
                    self.validate_function_declaration(
                        node, node_idx, scope_idx, symbol_table, level
                    )
                elif node_type in ["function_call", "function_call_assignment"]:
                    self.validate_function_call(
                        node, node_idx, scope_idx, symbol_table, level
                    )
                elif node_type == "return":
                    self.validate_return(node, node_idx, scope_idx, symbol_table, level)
                elif node_type in ["while_loop", "for_loop"]:
                    self.validate_loop_node(
                        node, node_idx, scope_idx, symbol_table, level
                    )

    def validate_node_types(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Валидирует типы в узле"""
        node_type = node.get("node", "")

        if node_type == "openmp_pragma":
            self.add_error(
                node.get("error", "OpenMP pragma must be attached to a for loop"),
                scope_idx,
                node_idx,
                source_line=node.get("source_line"),
            )
            return

        if node_type == "assignment":
            self.validate_assignment_types(node, node_idx, scope_idx, level)
        elif node_type == "declaration":
            self.validate_declaration_types(node, node_idx, scope_idx, level)
        elif node_type == "return":
            self.validate_return_types(node, node_idx, scope_idx, level)
        elif node_type == "while_loop":
            self.validate_while_condition_types(node, node_idx, scope_idx, level)
        elif node_type == "if_statement":
            self.validate_if_condition_types(node, node_idx, scope_idx, level)
        elif node_type in ["binary_operation", "unary_operation"]:
            self.validate_operation_types(node, node_idx, scope_idx, level)

    def validate_assignment_types(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Валидирует типы в присваивании"""
        symbols = node.get("symbols", [])
        expression_ast = node.get("expression_ast")

        if not symbols or not expression_ast:
            return

        target_var = symbols[0]
        target_info = self.get_symbol_info(target_var, level)

        if not target_info:
            return

        target_type = target_info.get("type", "")
        value_type = self.get_type_from_ast(expression_ast, scope_idx, node_idx, level)

        # Проверяем совместимость типов
        if not self.are_types_compatible(target_type, value_type):
            self.add_error(
                f"нельзя присвоить значение типа '{value_type}' переменной типа '{target_type}'",
                scope_idx,
                node_idx,
            )

        # Рекурсивно проверяем типы в AST
        self.validate_ast_types(expression_ast, node_idx, scope_idx, level)

    def validate_declaration_types(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Проверяет тип объявления по типизированному AST."""
        var_type = node.get("var_type", "")
        expression_ast = node.get("expression_ast")

        if not var_type or not expression_ast:
            return

        value_type = self.get_type_from_ast(expression_ast, scope_idx, node_idx, level)
        if value_type == "unknown":
            return

        if not self.are_types_compatible(var_type, value_type):
            self.add_error(
                f"нельзя присвоить значение типа '{value_type}' переменной типа '{var_type}'",
                scope_idx,
                node_idx,
            )

        self.validate_ast_types(expression_ast, node_idx, scope_idx, level)

    def validate_return_types(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Валидирует типы возвращаемых значений"""
        return
        # Получаем текущую функцию
        current_scope = self.get_scope_by_level(level)
        if not current_scope or current_scope.get("type") != "function":
            return

        declared_return_type = current_scope.get("return_type", "None")
        return_value_ast = node.get("operations", [{}])[0].get("value")

        if not return_value_ast:
            return

        actual_return_type = self.get_type_from_ast(
            return_value_ast, scope_idx, node_idx, level
        )

        if not self.are_types_compatible(declared_return_type, actual_return_type):
            self.add_error(
                f"функция объявлена как возвращающая '{declared_return_type}', "
                f"но возвращает значение типа '{actual_return_type}'",
                scope_idx,
                node_idx,
            )

    def validate_while_condition_types(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Валидирует типы в условии while"""
        condition_ast = node.get("condition_ast")

        if condition_ast:
            condition_type = self.get_type_from_ast(
                condition_ast, scope_idx, node_idx, level
            )

            # Условие должно быть bool
            if condition_type not in ["bool", "unknown"] and condition_type != "bool":
                self.add_error(
                    f"условие цикла while должно быть bool, получено: {condition_type}",
                    scope_idx,
                    node_idx,
                )

            # Рекурсивно проверяем AST условия
            self.validate_ast_types(condition_ast, node_idx, scope_idx, level)

    def validate_if_condition_types(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Валидирует типы в условии if/elif"""
        condition_ast = node.get("condition_ast")

        if condition_ast:
            condition_type = self.get_type_from_ast(
                condition_ast, scope_idx, node_idx, level
            )

            # Условие должно быть bool
            if condition_type not in ["bool", "unknown"] and condition_type != "bool":
                self.add_error(
                    f"условие if должно быть bool, получено: {condition_type}",
                    scope_idx,
                    node_idx,
                )

            # Рекурсивно проверяем AST условия
            self.validate_ast_types(condition_ast, node_idx, scope_idx, level)

        # Проверяем elif блоки
        for elif_block in node.get("elif_blocks", []):
            self.validate_if_condition_types(elif_block, node_idx, scope_idx, level)

    def validate_operation_types(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Валидирует типы в операциях"""
        operations = node.get("operations", [])

        for op in operations:
            op_type = op.get("type")

            if op_type == "BINARY_OPERATION":
                left_ast = op.get("left")
                right_ast = op.get("right")
                operator = op.get("operator_symbol", "")

                if left_ast and right_ast:
                    left_type = self.get_type_from_ast(
                        left_ast, scope_idx, node_idx, level
                    )
                    right_type = self.get_type_from_ast(
                        right_ast, scope_idx, node_idx, level
                    )

                    if not self.can_operate_between_types(
                        left_type, right_type, operator
                    ):
                        self.add_error(
                            f"нельзя выполнить операцию '{operator}' "
                            f"между типами '{left_type}' и '{right_type}'",
                            scope_idx,
                            node_idx,
                        )

            elif op_type == "UNARY_OPERATION":
                operand_ast = op.get("operand")
                operator = op.get("operator_symbol", "")

                if operand_ast:
                    operand_type = self.get_type_from_ast(
                        operand_ast, scope_idx, node_idx, level
                    )

                    if operator == "not" and operand_type != "bool":
                        self.add_error(
                            f"оператор 'not' применяется к типу '{operand_type}', а не к bool",
                            scope_idx,
                            node_idx,
                        )

    def get_parent_scope(self, level: int) -> Optional[Dict]:
        """Находит родительский scope для заданного уровня"""
        # Ищем scope с уровнем, указанным как parent_scope
        for scope in self.all_scopes:
            if scope.get("level") == level:
                parent_level = scope.get("parent_scope")
                if parent_level is not None:
                    # Ищем scope с таким уровнем
                    for parent_scope in self.all_scopes:
                        if parent_scope.get("level") == parent_level:
                            return parent_scope
        return None

    def validate_declaration(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует объявление переменной"""
        symbols = node.get("symbols", [])
        operations = node.get("operations", [])
        content = node.get("content", "")

        for symbol in symbols:
            if symbol not in symbol_table:
                self.add_error(
                    f"объявляемая переменная '{symbol}' отсутствует в symbol_table",
                    scope_idx,
                    node_idx,
                )
            else:
                for op in operations:
                    if op.get("type") in ["NEW_VAR", "NEW_CONST"]:
                        declared_type = op.get("var_type") or op.get("const_type")
                        actual_type = symbol_table[symbol].get("type")
                        if declared_type != actual_type:
                            self.add_error(
                                f"тип переменной '{symbol}' не совпадает (объявлен: {declared_type}, в symbol_table: {actual_type})",
                                scope_idx,
                                node_idx,
                            )

        # Проверяем значение инициализации
        if content:
            # Парсим объявление
            pattern = r"var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)"
            match = re.match(pattern, content)

            if match:
                var_name, var_type, value = match.groups()

                # Проверяем выражение инициализации
                self.validate_expression(value, scope_idx, node_idx, level)

                # Новые parser nodes carry expression_ast and are validated by
                # validate_declaration_types.  Keep the text fallback only for
                # old serialized graphs without an AST.
                if not node.get("expression_ast"):
                    self.validate_type_compatibility(
                        var_name, value, scope_idx, node_idx, level
                    )

    def validate_expression(
        self, expression: str, scope_idx: int, node_idx: int, level: int
    ):
        """Валидирует выражение (правая часть присваивания или инициализации)"""
        expression = expression.strip()

        constructor_pattern = r"([A-Z][a-zA-Z0-9_]*)\s*\(([^)]*)\)"
        match = re.match(constructor_pattern, expression)
        if match:
            class_name = match.group(1)
            # Проверяем, что класс существует
            if class_name not in self.classes:
                self.add_error(
                    f"класс '{class_name}' не объявлен",
                    scope_idx,
                    node_idx,
                )
            return

        # Сначала проверяем, не является ли это литералом
        if (expression.startswith('"') and expression.endswith('"')) or (
            expression.startswith("'") and expression.endswith("'")
        ):
            return

        if expression.isdigit() or (
            expression.startswith("-") and expression[1:].isdigit()
        ):
            return

        if expression in ["True", "False", "None"]:
            return

        # Проверяем операции с указателями
        if expression.startswith("&"):
            # Адрес переменной
            var_name = expression[1:].strip()
            if var_name and var_name.isalpha():
                if not self.find_symbol_in_scope(var_name, level):
                    self.add_error(
                        f"переменная '{var_name}' для взятия адреса не объявлена",
                        scope_idx,
                        node_idx,
                    )
            return

        elif expression.startswith("*"):
            # Разыменование указателя
            pointer_name = expression[1:].strip()
            if pointer_name and pointer_name.isalpha():
                pointer_info = self.get_symbol_info(pointer_name, level)
                if not pointer_info:
                    self.add_error(
                        f"указатель '{pointer_name}' для разыменования не найден",
                        scope_idx,
                        node_idx,
                    )
                elif not pointer_info.get("type", "").startswith("*"):
                    self.add_error(
                        f"переменная '{pointer_name}' не является указателем",
                        scope_idx,
                        node_idx,
                    )
            return

        # Never interpret identifiers inside quoted literals as symbols.
        # Examples: "./examples/ocean.json", "foo(bar)", "user.name".
        # The parser has already represented strings as literal AST nodes;
        # this legacy textual validator must ignore their contents.
        scan_expression = re.sub(
            r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
            "",
            expression,
        )

        # Проверяем вызовы функций - ИГНОРИРУЕМ функции с @
        func_calls = re.findall(
            r"(@?[a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            scan_expression,
        )
        for func_name in func_calls:
            # Игнорируем функции, начинающиеся с @
            if func_name.startswith("@"):
                logger.debug(
                    f"    Пропускаем проверку функции '{func_name}' (игнорируемая)"
                )
                continue

            # ``object.method(...)`` is validated by the method/type pass;
            # the textual fallback must not treat ``method`` as a free
            # function.
            if re.search(rf"\.\s*{re.escape(func_name)}\s*\(", expression):
                continue

            if (
                func_name not in self.functions
                and func_name not in self.builtin_functions
                and func_name not in self.external_c_functions
            ):
                self.add_error(
                    f"функция '{func_name}' не объявлена", scope_idx, node_idx
                )

        # Проверяем переменные (игнорируя части внутри вызовов функций)
        # Убираем все вызовы функций для упрощения
        temp_expr = scan_expression
        for func_name in func_calls:
            # Простое удаление вызовов функций
            temp_expr = temp_expr.replace(f"{func_name}(", "")

        # Attribute names (for example ``tensor.shape``) are not standalone
        # variables.  Keep only the receiver for declaration checks.
        temp_expr = re.sub(r"\.[a-zA-Z_][a-zA-Z0-9_]*", "", temp_expr)

        # Ищем переменные
        var_pattern = r'(?<!["\'])(?<![a-zA-Z0-9_])\b([a-zA-Z_][a-zA-Z0-9_]+)\b(?![a-zA-Z0-9_])(?!["\'])'
        identifiers = re.findall(var_pattern, temp_expr)

        for identifier in identifiers:
            # Пропускаем ключевые слова, типы данных, литералы
            if identifier in ["True", "False", "None"] or identifier.isdigit():
                continue

            # Пропускаем функции с @
            if identifier.startswith("@"):
                continue

            # Проверяем, не является ли это вызовом функции (уже обработали)
            if identifier in func_calls:
                continue

            # Это переменная - проверяем ее существование
            logger.debug(f"      Проверка переменной '{identifier}' в выражении")
            if not self.find_symbol_in_scope(identifier, level):
                self.add_error(
                    f"переменная '{identifier}' в выражении не объявлена",
                    scope_idx,
                    node_idx,
                )
            elif self.is_variable_deleted(identifier, level):
                self.add_error(
                    f"переменная '{identifier}' в выражении была удалена",
                    scope_idx,
                    node_idx,
                )

    def validate_delete(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует удаление переменной"""
        symbols = node.get("symbols", [])

        for symbol in symbols:
            # Проверяем, существует ли переменная
            symbol_info = self.get_symbol_info(symbol, level)
            if not symbol_info:
                self.add_error(
                    f"удаляемая переменная '{symbol}' не объявлена", scope_idx, node_idx
                )
                continue

            # Проверяем, константа ли это
            if symbol_info.get("key") == "const":
                self.add_error(
                    f"попытка удаления константы '{symbol}'", scope_idx, node_idx
                )
                continue

            # Проверяем, не была ли уже удалена
            key = (level, symbol)
            current_state = self.variable_states.get(key)

            if current_state == "deleted":
                self.add_error(
                    f"переменная '{symbol}' уже была удалена", scope_idx, node_idx
                )
            else:
                # Помечаем как удаленную
                self.variable_states[key] = "deleted"
                logger.debug(f"    Переменная '{symbol}' помечена как удаленная")

    def validate_unary_operation(
        self, op: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Валидирует унарную операцию"""
        value = op.get("value")

        if value and value.isalpha() and value not in ["True", "False", "None"]:
            if not self.find_symbol_in_scope(value, level):
                self.add_error(
                    f"операнд унарной операции '{value}' не объявлен",
                    scope_idx,
                    node_idx,
                )
            elif self.is_variable_deleted(value, level):
                self.add_error(
                    f"операнд унарной операции '{value}' был удален",
                    scope_idx,
                    node_idx,
                )

    def validate_augmented_assignment(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует составное присваивание"""
        symbols = node.get("symbols", [])
        operations = node.get("operations", [])
        dependencies = node.get("dependencies", [])

        for symbol in symbols:
            symbol_info = self.get_symbol_info(symbol, level)
            if not symbol_info:
                self.add_error(
                    f"переменная '{symbol}' в составном присваивании не объявлена",
                    scope_idx,
                    node_idx,
                )
            else:
                # Проверяем, не была ли переменная удалена
                if self.is_variable_deleted(symbol, level):
                    self.add_error(
                        f"переменная '{symbol}' была удалена и требует переинициализации",
                        scope_idx,
                        node_idx,
                    )
                elif symbol_info.get("key") == "const":
                    self.add_error(
                        f"попытка модификации константы '{symbol}'", scope_idx, node_idx
                    )

        for dep in dependencies:
            if not self.find_symbol_in_scope(dep, level):
                self.add_error(
                    f"используемая переменная '{dep}' не объявлена", scope_idx, node_idx
                )
            elif self.is_variable_deleted(dep, level):
                self.add_error(
                    f"используемая переменная '{dep}' была удалена", scope_idx, node_idx
                )

    def validate_function_declaration(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует объявление функции"""
        func_name = node.get("function_name")
        parameters = node.get("parameters", [])
        is_stub = node.get("is_stub", False)

        # Проверяем дублирование функций
        for other_scope in self.all_scopes:
            if other_scope.get("level") <= level:
                for other_node in other_scope.get("graph", []):
                    if (
                        other_node.get("node") == "function_declaration"
                        and other_node.get("function_name") == func_name
                        and other_node is not node
                    ):
                        self.add_error(
                            f"функция '{func_name}' уже объявлена", scope_idx, node_idx
                        )
                        return

        # Проверяем параметры
        param_names = set()
        for param in parameters:
            param_name = param.get("name")
            if param_name in param_names:
                self.add_error(
                    f"дублирующийся параметр '{param_name}' в функции '{func_name}'",
                    scope_idx,
                    node_idx,
                )
            param_names.add(param_name)

        # Если это заглушка, выводим предупреждение
        if is_stub:
            self.add_warning(
                f"функция '{func_name}' объявлена как заглушка (только pass)",
                scope_idx,
                node_idx,
            )

    def validate_function_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов функции с поддержкой AST аргументов"""
        func_name = node.get("function")
        arguments = node.get("arguments", [])

        # ИГНОРИРОВАНИЕ: если функция начинается с @ - пропускаем стандартные проверки
        if func_name and func_name.startswith("@"):
            logger.debug(
                f"✓ Вызов функции '{func_name}' - пропускаем стандартную проверку (C-code/игнорируемая функция)"
            )

            # Только базовая проверка аргументов (если нужно)
            for arg in arguments:
                self._validate_argument(arg, func_name, scope_idx, node_idx, level)
            return  # Завершаем проверку для этой функции

        # Проверяем, что функция существует или является встроенной
        if (
            func_name not in self.functions
            and func_name not in self.builtin_functions
            and func_name not in self.external_c_functions
        ):
            self.add_error(
                f"вызываемая функция '{func_name}' не объявлена", scope_idx, node_idx
            )
        elif func_name in self.functions:
            func_info = self.functions[func_name]
            func_params = func_info.get("parameters", [])

            if len(arguments) != len(func_params):
                self.add_error(
                    f"функция '{func_name}' ожидает {len(func_params)} аргументов, передано {len(arguments)}",
                    scope_idx,
                    node_idx,
                )

        # Проверяем аргументы
        for arg in arguments:
            self._validate_argument(arg, func_name, scope_idx, node_idx, level)

    def _validate_argument(
        self, arg, func_name: str, scope_idx: int, node_idx: int, level: int
    ):
        """Валидирует один аргумент (может быть строкой или AST)"""
        if not arg:
            return

        # Если аргумент - строка
        if isinstance(arg, str):
            # Пропускаем NULL, True, False, None
            if arg in ["NULL", "True", "False", "None"]:
                return

            # Пропускаем литералы
            if (arg.startswith('"') and arg.endswith('"')) or (
                arg.startswith("'") and arg.endswith("'")
            ):
                return

            # Пропускаем числа
            if arg.isdigit() or (arg.startswith("-") and arg[1:].isdigit()):
                return

            # Пропускаем вызовы конструкторов
            if "(" in arg:
                # Это вызов функции или конструктора
                return

            # Проверяем обычные переменные
            if arg.isalpha():
                if not self.find_symbol_in_scope(arg, level):
                    self.add_error(f"аргумент '{arg}' не объявлен", scope_idx, node_idx)
                elif self.is_variable_deleted(arg, level):
                    self.add_error(f"аргумент '{arg}' был удален", scope_idx, node_idx)

        # Если аргумент - AST (словарь)
        elif isinstance(arg, Mapping):
            arg_type = arg.get("type")

            # Пропускаем конструкторы классов
            if arg_type == "constructor_call":
                return

            # Пропускаем литералы
            if arg_type == "literal":
                return

            # Извлекаем зависимости из AST
            dependencies = self._extract_dependencies_from_ast(arg)

            for dep in dependencies:
                if not self.find_symbol_in_scope(dep, level):
                    self.add_error(
                        f"переменная '{dep}' в аргументе функции '{func_name}' не объявлена",
                        scope_idx,
                        node_idx,
                    )
                elif self.is_variable_deleted(dep, level):
                    self.add_error(
                        f"переменная '{dep}' в аргументе функции '{func_name}' была удалена",
                        scope_idx,
                        node_idx,
                    )

    def _extract_dependencies_from_ast(self, ast: Dict) -> List[str]:
        """Извлекает зависимости (имена переменных) из AST"""
        dependencies = []

        def traverse(node):
            if not isinstance(node, Mapping):
                return

            node_type = node.get("type")

            if node_type == "variable":
                var_name = node.get("name") or node.get("value")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)

            elif node_type == "attribute_access":
                obj_name = node.get("object")
                if obj_name and obj_name not in dependencies:
                    dependencies.append(obj_name)

            elif node_type == "method_call":
                obj_name = node.get("object")
                if obj_name and obj_name not in dependencies:
                    dependencies.append(obj_name)

                for arg in node.get("arguments", []):
                    traverse(arg)

            elif node_type == "constructor_call":
                for arg in node.get("arguments", []):
                    traverse(arg)

            elif node_type == "function_call":
                # Пользовательские функции добавляем как зависимости
                func_name = node.get("function")
                if func_name and func_name not in self.builtin_functions:
                    if func_name not in dependencies:
                        dependencies.append(func_name)

                for arg in node.get("arguments", []):
                    traverse(arg)

            elif node_type == "binary_operation":
                traverse(node.get("left"))
                traverse(node.get("right"))

            elif node_type == "unary_operation":
                traverse(node.get("operand"))

            elif node_type == "ternary_operator":
                traverse(node.get("condition"))
                traverse(node.get("true_expr"))
                traverse(node.get("false_expr"))

            elif node_type == "list_literal":
                for item in node.get("items", []):
                    traverse(item)

            elif node_type == "tuple_literal":
                for item in node.get("items", []):
                    traverse(item)

            elif node_type == "dict_literal":
                for value in node.get("pairs", {}).values():
                    traverse(value)

            elif node_type == "set_literal":
                for item in node.get("items", []):
                    traverse(item)

            elif node_type == "address_of":
                var_name = node.get("variable")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)

            elif node_type == "dereference":
                pointer_name = node.get("pointer")
                if pointer_name and pointer_name not in dependencies:
                    dependencies.append(pointer_name)

            elif node_type == "index_access":
                var_name = node.get("variable")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)
                traverse(node.get("index"))

            elif node_type == "slice_access":
                var_name = node.get("variable")
                if var_name and var_name not in dependencies:
                    dependencies.append(var_name)

                start = node.get("start")
                stop = node.get("stop")
                step = node.get("step")

                if start:
                    traverse(start)
                if stop:
                    traverse(stop)
                if step:
                    traverse(step)

        traverse(ast)
        return dependencies

    def validate_print(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов функции print"""
        arguments = node.get("arguments", [])
        dependencies = node.get("dependencies", [])

        for dep in dependencies:
            if not self.find_symbol_in_scope(dep, level):
                self.add_error(
                    f"переменная '{dep}' в аргументе print не объявлена",
                    scope_idx,
                    node_idx,
                )
            elif self.is_variable_deleted(dep, level):
                self.add_error(
                    f"переменная '{dep}' в аргументе print была удалена",
                    scope_idx,
                    node_idx,
                )

        # Дополнительно проверяем аргументы для сложных выражений
        for arg in arguments:
            if (
                arg
                and not arg.startswith('"')
                and not arg.endswith('"')
                and not arg.startswith("'")
                and not arg.endswith("'")
                and not arg.isdigit()
                and arg not in ["True", "False", "None"]
            ):
                # Ищем переменные в сложных выражениях
                var_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)"
                vars_in_arg = re.findall(var_pattern, arg)
                for var in vars_in_arg:
                    if (
                        var not in ["True", "False", "None"]
                        and var not in self.builtin_functions
                        and not self.find_symbol_in_scope(var, level)
                    ):
                        self.add_error(
                            f"переменная '{var}' в выражении print не объявлена",
                            scope_idx,
                            node_idx,
                        )
                    elif self.is_variable_deleted(var, level):
                        self.add_error(
                            f"переменная '{var}' в выражении print была удалена",
                            scope_idx,
                            node_idx,
                        )

    def validate_len_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов len()"""
        arguments = node.get("arguments", [])

        if len(arguments) != 1:
            return

        arg = arguments[0]

        # len() принимает строки или массивы (пока только строки)
        if arg.startswith('"') and arg.endswith('"'):
            return

        if arg.startswith("'") and arg.endswith("'"):
            return

        # Для переменных нужно проверить тип
        if arg.isalpha():
            symbol_info = self.get_symbol_info(arg, level)
            if symbol_info:
                var_type = symbol_info.get("type")
                if var_type not in ["str", "list", "array"]:
                    self.add_error(
                        f"функция len() ожидает строку, передана переменная типа '{var_type}'",
                        scope_idx,
                        node_idx,
                    )

    def validate_int_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов int()"""
        arguments = node.get("arguments", [])

        if len(arguments) != 1:
            return

        arg = arguments[0]

        # int() принимает числа, строки с числами или bool
        if arg.isdigit() or arg in ["True", "False"]:
            return

        if arg.startswith('"') and arg.endswith('"'):
            str_value = arg[1:-1]
            if not str_value.lstrip("-").isdigit():
                self.add_error(
                    f"функция int() не может преобразовать строку '{arg}' в число",
                    scope_idx,
                    node_idx,
                )

        # Для переменных проверяем тип
        if arg.isalpha():
            symbol_info = self.get_symbol_info(arg, level)
            if symbol_info:
                var_type = symbol_info.get("type")
                if var_type not in ["int", "str", "bool"]:
                    self.add_error(
                        f"функция int() ожидает int, string или bool, передана переменная типа '{var_type}'",
                        scope_idx,
                        node_idx,
                    )

    def validate_str_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов str()"""
        arguments = node.get("arguments", [])

        if len(arguments) != 1:
            return

        # str() принимает любые значения
        # Проверяем только, что аргумент существует
        arg = arguments[0]
        if arg.isalpha() and not self.find_symbol_in_scope(arg, level):
            self.add_error(
                f"переменная '{arg}' в аргументе str() не объявлена",
                scope_idx,
                node_idx,
            )
        elif arg.isalpha() and self.is_variable_deleted(arg, level):
            self.add_error(
                f"переменная '{arg}' в аргументе str() была удалена",
                scope_idx,
                node_idx,
            )

    def validate_bool_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов bool()"""
        arguments = node.get("arguments", [])

        if len(arguments) != 1:
            return

        # bool() принимает любые значения
        # Проверяем только, что аргумент существует
        arg = arguments[0]
        if arg.isalpha() and not self.find_symbol_in_scope(arg, level):
            self.add_error(
                f"переменная '{arg}' в аргументе bool() не объявлена",
                scope_idx,
                node_idx,
            )
        elif arg.isalpha() and self.is_variable_deleted(arg, level):
            self.add_error(
                f"переменная '{arg}' в аргументе bool() была удалена",
                scope_idx,
                node_idx,
            )

    def validate_range_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов range()"""
        arguments = node.get("arguments", [])

        # range() принимает 1-3 аргумента, все должны быть int
        for i, arg in enumerate(arguments):
            if arg.isdigit():
                continue

            if arg.isalpha():
                symbol_info = self.get_symbol_info(arg, level)
                if symbol_info:
                    var_type = symbol_info.get("type")
                    if var_type != "int":
                        self.add_error(
                            f"аргумент {i + 1} функции range() должен быть int, передана переменная типа '{var_type}'",
                            scope_idx,
                            node_idx,
                        )
                else:
                    self.add_error(
                        f"переменная '{arg}' в аргументе range() не объявлена",
                        scope_idx,
                        node_idx,
                    )

                # Проверяем, не удалена ли переменная
                if self.is_variable_deleted(arg, level):
                    self.add_error(
                        f"переменная '{arg}' в аргументе range() была удалена",
                        scope_idx,
                        node_idx,
                    )

    def validate_return(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует оператор return"""
        dependencies = node.get("dependencies", [])
        content = node.get("content", "")
        operations = node.get("operations", [])

        # Парсим return выражение
        if content.startswith("return "):
            return_expr = content[7:].strip()  # Убираем "return "
            logger.debug(f"  Проверка return: '{return_expr}'")

            # Получаем текущий scope (функцию) - КОРРЕКТНО
            current_scope = self.get_scope_for_node(scope_idx, level)

            if not current_scope or current_scope.get("type") != "function":
                logger.debug(f"    Return не в функции, пропускаем проверку типа")
                return

            declared_return_type = current_scope.get("return_type", "None")
            logger.debug(
                f"    Функция объявлена как возвращающая: {declared_return_type}"
            )

            # Получаем тип возвращаемого значения
            actual_return_type = "unknown"

            # Используем AST из operations если есть
            for op in operations:
                if op.get("type") == "RETURN":
                    value_ast = op.get("value")
                    if value_ast:
                        actual_return_type = self.get_type_from_ast(
                            value_ast, scope_idx, node_idx, level
                        )
                        logger.debug(
                            f"    Тип возвращаемого значения из AST: {actual_return_type}"
                        )
                        break

            # Если не нашли в AST, пытаемся определить из выражения
            if actual_return_type == "unknown":
                actual_return_type = self.guess_type_from_value(return_expr)
                logger.debug(
                    f"    Тип возвращаемого значения из выражения: {actual_return_type}"
                )

            # Проверяем совместимость типов
            if actual_return_type != "unknown" and not self.are_types_compatible(
                declared_return_type, actual_return_type
            ):
                self.add_error(
                    f"функция объявлена как возвращающая '{declared_return_type}', "
                    f"фактически возвращает '{actual_return_type}'",
                    scope_idx,
                    node_idx,
                )

            # Проверяем сложные выражения
            self.validate_expression(return_expr, scope_idx, node_idx, level)

        # Старая проверка dependencies (для совместимости)
        for dep in dependencies:
            if not self.find_symbol_in_scope(dep, level):
                self.add_error(
                    f"возвращаемая переменная '{dep}' не объявлена", scope_idx, node_idx
                )
            elif self.is_variable_deleted(dep, level):
                self.add_error(
                    f"возвращаемая переменная '{dep}' была удалена", scope_idx, node_idx
                )

    def validate_loop_node(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует узел цикла"""
        node_type = node.get("node")

        if node_type == "while_loop":
            condition = node.get("condition", {})
            if condition.get("type") == "COMPARISON":
                left = condition.get("left")
                right = condition.get("right")

                for var in [left, right]:
                    if var and var.isalpha() and var not in ["True", "False", "None"]:
                        if not self.find_symbol_in_scope(var, level):
                            self.add_error(
                                f"переменная '{var}' в условии цикла не объявлена",
                                scope_idx,
                                node_idx,
                            )
                        elif self.is_variable_deleted(var, level):
                            self.add_error(
                                f"переменная '{var}' в условии цикла была удалена",
                                scope_idx,
                                node_idx,
                            )

        elif node_type == "for_loop":
            loop_var = node.get("loop_variable")
            iterable = node.get("iterable", {})

            if loop_var not in symbol_table:
                self.add_error(
                    f"переменная цикла '{loop_var}' не объявлена", scope_idx, node_idx
                )
            elif self.is_variable_deleted(loop_var, level):
                self.add_error(
                    f"переменная цикла '{loop_var}' была удалена", scope_idx, node_idx
                )

            if iterable.get("type") == "RANGE_CALL":
                args = iterable.get("arguments", {})
                for arg_name, arg_value in args.items():
                    if (
                        arg_value
                        and arg_value.isalpha()
                        and arg_value not in ["True", "False", "None"]
                    ):
                        if not self.find_symbol_in_scope(arg_value, level):
                            self.add_error(
                                f"аргумент range '{arg_value}' не объявлен",
                                scope_idx,
                                node_idx,
                            )
                        elif self.is_variable_deleted(arg_value, level):
                            self.add_error(
                                f"аргумент range '{arg_value}' был удален",
                                scope_idx,
                                node_idx,
                            )

            if node.get("openmp"):
                self.validate_openmp_loop(node, node_idx, scope_idx, level)

    def _openmp_clause_arguments(self, metadata: Dict) -> dict[str, list[str]]:
        """Return OpenMP clauses grouped by name, preserving duplicates."""
        grouped: dict[str, list[str]] = {}
        for clause in metadata.get("clauses", []) or []:
            name = str(clause.get("name", "")).strip()
            arguments = str(clause.get("arguments", "")).strip()
            grouped.setdefault(name, []).append(arguments)
        return grouped

    def _openmp_csv_identifiers(self, arguments: str) -> list[str] | None:
        values = [item.strip() for item in arguments.split(",") if item.strip()]
        if not values or any(not re.match(r"^[A-Za-z_]\w*$", item) for item in values):
            return None
        return values

    def _openmp_reduction_arguments(self, arguments: str) -> tuple[str, list[str]] | None:
        match = re.match(r"^([+\-*\/%]|&&|\|\||max|min)\s*:\s*(.+)$", arguments)
        if not match:
            return None
        variables = self._openmp_csv_identifiers(match.group(2))
        if variables is None:
            return None
        return match.group(1), variables

    def _openmp_indices_contain_loop_vars(self, indices, loop_vars: set[str]) -> bool:
        names = set()
        if isinstance(indices, list):
            for index in indices:
                if isinstance(index, Mapping):
                    self._collect_vars_from_ast(index, names)
        elif isinstance(indices, Mapping):
            self._collect_vars_from_ast(indices, names)
        elif isinstance(indices, str):
            names.add(indices)
        return loop_vars.issubset(names)

    def _openmp_scalar_type(self, type_name: str) -> bool:
        """Whether a type is safe to create/use as a private scalar."""
        normalized = (type_name or "").strip()
        while normalized.startswith("&mut ") or normalized.startswith("&"):
            normalized = normalized[1:].strip()
            if normalized.startswith("mut "):
                normalized = normalized[4:].strip()
        return normalized in {
            "bool",
            "int",
            "float",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        }

    def _validate_openmp_body(
        self,
        body: list[Dict],
        loop_var: str,
        level: int,
        reduction_vars: set[str],
        private_vars: set[str],
        scope_idx: int,
        node_idx: int,
        loop_vars: set[str] | None = None,
    ) -> None:
        """Validate the deliberately conservative, race-aware OpenMP subset."""
        active_loop_vars = set(loop_vars or {loop_var})
        declared_private = set()
        for body_node in body:
            if body_node.get("node") in {"declaration", "redeclaration"}:
                declared_private.update(body_node.get("symbols", []))

        forbidden = {
            "break",
            "continue",
            "return",
            "while_loop",
            "function_call",
            "function_call_assignment",
            "builtin_function_call",
            "builtin_function_call_assignment",
            "method_call",
            "static_method_call",
            "c_call",
            "attribute_assignment",
            "slice_assignment",
            "delete",
        }

        for body_node in body:
            body_type = body_node.get("node", "")
            if body_type in forbidden:
                self.add_error(
                    f"узел '{body_type}' запрещен внутри OpenMP parallel for; "
                    "требуется потокобезопасная аннотация",
                    scope_idx,
                    node_idx,
                )
                continue

            if body_type in {"declaration", "redeclaration"}:
                var_type = body_node.get("var_type", "")
                if not self._openmp_scalar_type(var_type):
                    self.add_error(
                        f"тип '{var_type}' нельзя объявлять внутри OpenMP parallel for; "
                        "управляемые объекты пока не являются потокобезопасными",
                        scope_idx,
                        node_idx,
                    )
                continue

            if body_type in {"assignment", "augmented_assignment"}:
                targets = set(body_node.get("symbols", []))
                allowed_private = declared_private | private_vars
                if body_type == "augmented_assignment":
                    allowed_private |= reduction_vars
                invalid_targets = targets - allowed_private
                if invalid_targets:
                    self.add_error(
                        "запись в общую scalar-переменную внутри OpenMP loop "
                        f"без reduction/private: {', '.join(sorted(invalid_targets))}",
                        scope_idx,
                        node_idx,
                    )
                continue

            if body_type in {
                "index_assignment",
                "nested_index_assignment",
                "augmented_index_assignment",
            }:
                variable = body_node.get("variable", "")
                info = self.get_symbol_info(variable, level)
                var_type = info.get("type", "") if info else ""
                normalized_type = var_type.replace("&mut ", "").replace("&", "").strip()
                indices = body_node.get("indices")
                if indices is None:
                    indices = body_node.get("index")
                if not normalized_type.startswith(("array[", "Tensor")):
                    self.add_error(
                        f"запись в '{variable}' внутри OpenMP loop разрешена только для "
                        "array/Tensor",
                        scope_idx,
                        node_idx,
                    )
                elif not self._openmp_indices_contain_loop_vars(indices, active_loop_vars):
                    self.add_error(
                        f"индекс записи в '{variable}' не содержит все переменные "
                        f"вложенного цикла ({', '.join(sorted(active_loop_vars))}), "
                        "возможна гонка данных",
                        scope_idx,
                        node_idx,
                    )
                continue

            if body_type == "if_statement":
                nested = list(body_node.get("body", []))
                nested.extend(
                    item.get("body", []) for item in body_node.get("elif_blocks", [])
                )
                else_block = body_node.get("else_block") or {}
                nested.extend(else_block.get("body", []))
                self._validate_openmp_body(
                    nested,
                    loop_var,
                    level,
                    reduction_vars,
                    private_vars,
                    scope_idx,
                    node_idx,
                    active_loop_vars,
                )
                continue

            if body_type == "for_loop":
                # A non-OpenMP loop after the collapsed nest is sequential
                # inside each OpenMP iteration.  It must still be validated,
                # but it is not another parallel dimension.
                if body_node.get("openmp"):
                    self.add_error(
                        "отдельный OpenMP pragma внутри уже параллельного цикла "
                        "пока не поддерживается",
                        scope_idx,
                        node_idx,
                    )
                    continue
                if body_node.get("iterable", {}).get("type") != "RANGE_CALL":
                    self.add_error(
                        "последовательный вложенный цикл внутри OpenMP loop "
                        "должен использовать range(...)",
                        scope_idx,
                        node_idx,
                    )
                    continue
                nested_loop_var = body_node.get("loop_variable", "")
                self._validate_openmp_body(
                    body_node.get("body", []) or [],
                    nested_loop_var,
                    level,
                    reduction_vars,
                    private_vars | declared_private,
                    scope_idx,
                    node_idx,
                    active_loop_vars,
                )

    def _openmp_constant_step(self, node: Dict) -> bool:
        step = str(
            (node.get("iterable", {}).get("arguments", {}) or {}).get("step", "1")
        ).strip()
        return bool(re.match(r"^[+-]?[0-9]+$", step)) and int(step) != 0

    def _openmp_collapse_chain(
        self,
        node: Dict,
        count: int,
        scope_idx: int,
        node_idx: int,
    ) -> list[Dict] | None:
        """Validate and return the perfectly nested loop chain."""
        chain = [node]
        current = node
        for depth in range(1, count):
            body = current.get("body", []) or []
            if len(body) != 1 or body[0].get("node") != "for_loop":
                self.add_error(
                    f"collapse({count}) требует идеально вложенные for-циклы "
                    f"без промежуточных операторов (уровень {depth + 1})",
                    scope_idx,
                    node_idx,
                )
                return None
            current = body[0]
            if current.get("openmp"):
                self.add_error(
                    "внутренний цикл collapse не должен иметь отдельный OpenMP pragma",
                    scope_idx,
                    node_idx,
                )
                return None
            if current.get("iterable", {}).get("type") != "RANGE_CALL":
                self.add_error(
                    "каждый цикл в collapse должен использовать range(...)",
                    scope_idx,
                    node_idx,
                )
                return None
            if not self._openmp_constant_step(current):
                self.add_error(
                    "каждый цикл в collapse должен иметь постоянный ненулевой "
                    "целочисленный шаг",
                    scope_idx,
                    node_idx,
                )
                return None
            chain.append(current)
        return chain

    def validate_openmp_loop(self, node: Dict, node_idx: int, scope_idx: int, level: int):
        """Validate structured OpenMP metadata before C code generation."""
        metadata = node.get("openmp") or {}
        if metadata.get("error"):
            self.add_error(metadata["error"], scope_idx, node_idx)
            return
        if metadata.get("backend") != "openmp" or metadata.get("directive") != "parallel for":
            self.add_error("поддерживается только '#pragma omp parallel for'", scope_idx, node_idx)
            return

        if node.get("iterable", {}).get("type") != "RANGE_CALL":
            self.add_error(
                "OpenMP parallel for требует цикл по range(...)", scope_idx, node_idx
            )
            return

        if not self._openmp_constant_step(node):
            self.add_error(
                "OpenMP parallel for пока требует постоянный ненулевой целочисленный шаг range",
                scope_idx,
                node_idx,
            )

        grouped = self._openmp_clause_arguments(metadata)
        allowed = {
            "schedule",
            "collapse",
            "reduction",
            "private",
            "firstprivate",
            "lastprivate",
            "shared",
            "default",
            "nowait",
            "ordered",
        }
        for name in grouped:
            if name not in allowed:
                self.add_error(f"неподдерживаемая OpenMP clause '{name}'", scope_idx, node_idx)

        for argument in grouped.get("schedule", []):
            if not re.match(r"^(static|dynamic|guided|runtime|auto)(?:\s*,\s*[^,]+)?$", argument):
                self.add_error(f"некорректная OpenMP clause schedule({argument})", scope_idx, node_idx)

        collapse_count = 1
        collapse_values = grouped.get("collapse", [])
        if len(collapse_values) > 1 or (
            collapse_values
            and not re.match(r"^[1-9][0-9]*$", collapse_values[0])
        ):
            self.add_error(
                "collapse требует ровно одного положительного целого",
                scope_idx,
                node_idx,
            )
        elif collapse_values:
            collapse_count = int(collapse_values[0])

        reduction_vars = set()
        for argument in grouped.get("reduction", []):
            reduction = self._openmp_reduction_arguments(argument)
            if reduction is None:
                self.add_error(f"некорректная OpenMP clause reduction({argument})", scope_idx, node_idx)
                continue
            _, variables = reduction
            reduction_vars.update(variables)
            for variable in variables:
                info = self.get_symbol_info(variable, level)
                if not info or not self._openmp_scalar_type(info.get("type", "")):
                    self.add_error(
                        f"reduction-переменная '{variable}' должна быть scalar",
                        scope_idx,
                        node_idx,
                    )

        private_vars = set()
        for clause_name in {"private", "firstprivate", "lastprivate", "shared"}:
            for argument in grouped.get(clause_name, []):
                variables = self._openmp_csv_identifiers(argument)
                if variables is None:
                    self.add_error(f"некорректная OpenMP clause {clause_name}({argument})", scope_idx, node_idx)
                    continue
                if clause_name != "shared":
                    private_vars.update(variables)
                for variable in variables:
                    if not self.find_symbol_in_scope(variable, level):
                        self.add_error(f"OpenMP variable '{variable}' не объявлена", scope_idx, node_idx)

        for clause_name in {"nowait", "ordered"}:
            for argument in grouped.get(clause_name, []):
                if argument:
                    self.add_error(f"OpenMP clause '{clause_name}' не принимает аргументы", scope_idx, node_idx)

        if collapse_count > 1:
            chain = self._openmp_collapse_chain(
                node, collapse_count, scope_idx, node_idx
            )
            if chain is not None:
                loop_vars = {item.get("loop_variable", "") for item in chain}
                final_loop = chain[-1]
                self._validate_openmp_body(
                    final_loop.get("body", []) or [],
                    final_loop.get("loop_variable", ""),
                    level,
                    reduction_vars,
                    private_vars,
                    scope_idx,
                    node_idx,
                    loop_vars,
                )
        else:
            self._validate_openmp_body(
                node.get("body", []) or [],
                node.get("loop_variable", ""),
                level,
                reduction_vars,
                private_vars,
                scope_idx,
                node_idx,
            )

    def validate_function_return(self, scope: Dict, scope_idx: int):
        """Проверяет, что функция имеет return если нужно"""
        return_info = scope.get("return_info", {})
        return_type = scope.get("return_type", "None")
        is_stub = scope.get("is_stub", False)

        # Если функция - заглушка, пропускаем проверку return
        if is_stub:
            if return_type != "None":
                self.add_warning(
                    f"функция-заглушка возвращает '{return_type}' но не имеет return",
                    scope_idx,
                    None,
                )
            return

        # Обычная проверка для не-заглушек
        if return_type != "None" and not return_info.get("has_return", False):
            func_content = ""
            for node_idx, node in enumerate(scope.get("graph", [])):
                if node.get("node") == "function_declaration":
                    func_content = node.get("content", "")
                    break

            if func_content:
                self.add_warning(
                    f"функция возвращает '{return_type}' но не имеет оператора return",
                    scope_idx,
                    None,
                )

    def validate_loops(self, scope: Dict, scope_idx: int):
        """Проверяет циклы на корректность"""
        graph = scope.get("graph", [])

        for node_idx, node in enumerate(graph):
            if node.get("node") in ["while_loop", "for_loop"]:
                body = node.get("body", [])
                if not body:
                    self.add_warning(f"тело цикла пустое", scope_idx, node_idx)

    def validate_pointer_declaration(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует объявление указателя"""
        content = node.get("content", "")
        symbols = node.get("symbols", [])
        operations = node.get("operations", [])

        if not symbols:
            return

        pointer_name = symbols[0]
        pointer_info = self.get_symbol_info(pointer_name, level)

        if not pointer_info:
            return

        # Проверяем, что тип действительно указатель
        if not pointer_info.get("type", "").startswith("*"):
            self.add_error(
                f"переменная '{pointer_name}' объявлена как указатель, но тип не начинается с '*'",
                scope_idx,
                node_idx,
            )
            return

        # Получаем тип, на который указывает указатель
        pointed_type = pointer_info.get("type")[1:]  # Убираем звездочку

        # Проверяем операции с указателем
        for op in operations:
            if op.get("type") == "GET_ADDRESS":
                pointed_var = op.get("of")
                pointed_var_info = self.get_symbol_info(pointed_var, level)

                if pointed_var_info:
                    # Проверяем совместимость типов
                    pointed_var_type = pointed_var_info.get("type")
                    if pointed_var_type != pointed_type:
                        self.add_error(
                            f"указатель '*{pointed_type}' не может указывать на переменную '{pointed_var}' типа '{pointed_var_type}'",
                            scope_idx,
                            node_idx,
                        )

    def validate_dereference_write(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует запись через указатель (*p = значение)"""
        content = node.get("content", "")
        symbols = node.get("symbols", [])
        operations = node.get("operations", [])

        if not symbols:
            return

        pointer_name = symbols[0]
        pointer_info = self.get_symbol_info(pointer_name, level)

        if not pointer_info:
            self.add_error(f"указатель '{pointer_name}' не найден", scope_idx, node_idx)
            return

        # Проверяем, что это действительно указатель
        pointer_type = pointer_info.get("type", "")
        if not pointer_type.startswith("*"):
            self.add_error(
                f"переменная '{pointer_name}' не является указателем",
                scope_idx,
                node_idx,
            )
            return

        # Получаем тип, на который указывает указатель
        pointed_type = pointer_type[1:]  # Убираем звездочку

        # Получаем значение для присваивания
        for op in operations:
            if op.get("type") == "WRITE_POINTER":
                value = op.get("value", {})  # Теперь это AST

                # Получаем тип значения из AST
                value_type = self.get_type_from_ast(value, scope_idx, node_idx, level)

                if value_type and value_type != "unknown":
                    # Проверяем совместимость типов
                    if not self.are_types_compatible(pointed_type, value_type):
                        self.add_error(
                            f"нельзя присвоить значение типа '{value_type}' через указатель на '{pointed_type}'",
                            scope_idx,
                            node_idx,
                        )

    def get_type_from_ast(
        self, ast: Dict, scope_idx: int, node_idx: int, level: int
    ) -> str:
        """Определяет тип значения из AST"""
        if not isinstance(ast, Mapping):
            return "unknown"

        ast_type = ast.get("type")

        if ast_type == "literal":
            data_type = ast.get("data_type")
            if data_type:
                return data_type
            elif "value" in ast:
                val = ast["value"]
                if isinstance(val, str):
                    return "str"
                elif isinstance(val, int):
                    return "int"
                elif isinstance(val, bool):
                    return "bool"
                elif val is None:
                    return "None"

        elif ast_type == "variable":
            var_name = ast.get("value")
            if var_name:
                var_info = self.get_symbol_info(var_name, level)
                if var_info:
                    return var_info.get("type", "unknown")

        elif ast_type in {"attribute_access", "complex_attribute_access"}:
            attribute = ast.get("attribute", "")
            if attribute in {"length", "size", "ndim"}:
                return "int"
            if attribute == "shape" and ast_type == "complex_attribute_access":
                return "int"

        elif ast_type == "list_literal":
            return "list"

        elif ast_type in {"index_access", "tensor_index_access"}:
            variable = ast.get("variable")
            if variable:
                var_info = self.get_symbol_info(variable, level)
                if var_info:
                    var_type = var_info.get("type", "")
                    match = re.search(r"\[(.+)\]", var_type)
                    if match:
                        return match.group(1)
                    if ast_type == "index_access" and var_type == "str":
                        return "str"

        elif ast_type == "method_call":
            obj_info = self.get_symbol_info(ast.get("object", ""), level)
            obj_type = obj_info.get("type", "") if obj_info else ""

        elif ast_type == "binary_operation":
            # Определяем тип результата бинарной операции
            operator = ast.get("operator_symbol", "")
            left_type = self.get_type_from_ast(
                ast.get("left"), scope_idx, node_idx, level
            )
            right_type = self.get_type_from_ast(
                ast.get("right"), scope_idx, node_idx, level
            )

            # Для арифметических операций
            if operator in ["+", "-", "*", "/", "//", "%", "**"]:
                # String concatenation is the only arithmetic-style
                # operation currently supported for strings.
                if operator == "+" and left_type == "str" and right_type == "str":
                    return "str"
                if left_type == "float" or right_type == "float":
                    return "float"
                elif left_type == "int" and right_type == "int":
                    return "int"
                elif left_type == "unknown" or right_type == "unknown":
                    return "int"  # Предполагаем int по умолчанию

            # Для сравнений - возвращается bool
            elif operator in ["<", ">", "<=", ">=", "==", "!=", "and", "or"]:
                return "bool"

            return "int"  # По умолчанию для других операций

        elif ast_type == "function_call":
            func_name = ast.get("function")
            if func_name in self.builtin_functions:
                return self.builtin_functions[func_name]["return_type"]
            elif func_name in self.functions:
                func_info = self.functions[func_name]
                return func_info.get("return_type", "unknown")
            else:
                # Проверяем среди C функций
                return "unknown"  # Будет определено через guess_type_from_value

        elif ast_type == "dereference":
            pointer_name = ast.get("pointer")
            if pointer_name:
                pointer_info = self.get_symbol_info(pointer_name, level)
                if pointer_info:
                    pointer_type = pointer_info.get("type", "")
                    if pointer_type.startswith("*"):
                        return pointer_type[1:]  # Тип, на который указывает указатель

        return "unknown"

    def validate_assignment(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует присваивание"""
        symbols = node.get("symbols", [])
        dependencies = node.get("dependencies", [])
        content = node.get("content", "")
        expression_ast = node.get("expression_ast")

        # 1. Проверяем левую часть (целевую переменную)
        for symbol in symbols:
            symbol_info = self.get_symbol_info(symbol, level)

            if not symbol_info:
                self.add_error(
                    f"присваиваемая переменная '{symbol}' не объявлена",
                    scope_idx,
                    node_idx,
                )
            else:
                if self.is_variable_deleted(symbol, level):
                    # Проверяем переинициализацию
                    key = (level, symbol)
                    last_action = self.get_last_variable_action(symbol, level)
                    if last_action and last_action["action"] == "delete":
                        found_redeclaration = False
                        for action in self.variable_history.get(key, []):
                            if (
                                action["action"] == "declare"
                                and action["timestamp"] > last_action["timestamp"]
                            ):
                                found_redeclaration = True
                                break

                        if not found_redeclaration:
                            self.add_error(
                                f"переменная '{symbol}' была удалена и требует переинициализации",
                                scope_idx,
                                node_idx,
                            )
                elif symbol_info.get("key") == "const":
                    self.add_error(
                        f"попытка присваивания константе '{symbol}'",
                        scope_idx,
                        node_idx,
                    )

        # 2. Проверяем правую часть выражения
        if symbols and expression_ast:
            target_var = symbols[0]

            # Получаем тип значения из AST
            value_type = self.get_type_from_ast(
                expression_ast, scope_idx, node_idx, level
            )

            # Получаем тип целевой переменной
            target_info = self.get_symbol_info(target_var, level)
            if target_info:
                target_type = target_info.get("type", "")

                # Проверяем совместимость типов
                if value_type and value_type != "unknown" and target_type:
                    if not self.are_types_compatible(target_type, value_type):
                        self.add_error(
                            f"нельзя присвоить значение типа '{value_type}' переменной типа '{target_type}'",
                            scope_idx,
                            node_idx,
                        )

        # 3. Проверяем зависимости (используемые переменные)
        for dep in dependencies:
            if not self.find_symbol_in_scope(dep, level):
                self.add_error(
                    f"используемая переменная '{dep}' не объявлена", scope_idx, node_idx
                )
            elif self.is_variable_deleted(dep, level):
                self.add_error(
                    f"используемая переменная '{dep}' была удалена", scope_idx, node_idx
                )

    def validate_function_return_type(self, scope: Dict, scope_idx: int):
        """Проверяет соответствие типа возвращаемого значения"""
        return_info = scope.get("return_info", {})
        declared_return_type = scope.get("return_type", "None")

        if not return_info.get("has_return", False):
            # Функция не имеет return, но проверяем тип
            if declared_return_type != "None":
                self.add_warning(
                    f"функция объявлена как возвращающая '{declared_return_type}', но не имеет return",
                    scope_idx,
                    None,
                )
            return

        # Получаем информацию о возвращаемом значении
        return_value = return_info.get("return_value")
        if not return_value:
            return

        # Определяем фактический тип возвращаемого значения
        actual_return_type = self.determine_return_type(
            return_value, scope_idx, scope.get("level", 0)
        )

        if actual_return_type and actual_return_type != "unknown":
            # Сравниваем объявленный и фактический типы
            if not self.are_types_compatible(declared_return_type, actual_return_type):
                # Находим узел return в графе для правильной привязки ошибки
                graph = scope.get("graph", [])
                return_node_idx = -1

                for i, node in enumerate(graph):
                    if node.get("node") == "return":
                        return_node_idx = i
                        break

                # Добавляем ошибку только один раз
                if return_node_idx != -1:
                    self.add_error(
                        f"функция объявлена как возвращающая '{declared_return_type}', "
                        f"фактически возвращает '{actual_return_type}'",
                        scope_idx,
                        return_node_idx,
                    )

    def determine_return_type(self, return_value, scope_idx: int, level: int) -> str:
        """Определяет тип возвращаемого значения"""
        # Если return_value - строка (из content)
        if isinstance(return_value, str):
            # Парсим выражение
            if (return_value.startswith('"') and return_value.endswith('"')) or (
                return_value.startswith("'") and return_value.endswith("'")
            ):
                return "str"
            elif return_value.isdigit() or (
                return_value.startswith("-") and return_value[1:].isdigit()
            ):
                return "int"
            elif return_value in ["True", "False"]:
                return "bool"
            elif return_value == "None":
                return "None"
            else:
                # Это может быть переменная или выражение
                # Проверяем, не является ли это вызовом функции с @
                if "(" in return_value and ")" in return_value:
                    # Извлекаем имя функции
                    func_match = re.match(r"(@?[a-zA-Z_][a-zA-Z0-9_]*)\(", return_value)
                    if func_match:
                        func_name = func_match.group(1)

                        # Игнорируем функции с @
                        if func_name.startswith("@"):
                            logger.debug(
                                f"    Функция '{func_name}' игнорируется при определении типа возврата"
                            )
                            return "unknown"

                        # Получаем информацию о функции
                        func_info = None

                        # Проверяем в функциях
                        if func_name in self.functions:
                            func_info = self.functions[func_name]
                        else:
                            # Ищем в scope'ах
                            for scope in self.all_scopes:
                                if (
                                    scope.get("type") == "function"
                                    and scope.get("function_name") == func_name
                                ):
                                    return scope.get("return_type", "unknown")

                        if func_info:
                            return func_info.get("return_type", "unknown")

                # Если это переменная
                var_info = self.get_symbol_info(return_value, level)
                if var_info:
                    return var_info.get("type", "unknown")
                else:
                    return "unknown"

        # Если return_value - AST (словарь)
        elif isinstance(return_value, Mapping):
            return self.get_type_from_ast(return_value, scope_idx, None, level)

        return "unknown"

    def validate_type_compatibility(
        self, var_name: str, value: str, scope_idx: int, node_idx: int, level: int
    ):
        """Проверяет совместимость типов при присваивании"""
        var_info = self.get_symbol_info(var_name, level)
        if not var_info:
            return

        var_type = var_info.get("type")

        if "(" in value and ")" in value:
            # Извлекаем имя функции
            func_match = re.match(r"(@?[a-zA-Z_][a-zA-Z0-9_]*)\(", value)
            if func_match:
                func_name = func_match.group(1)

                # Игнорируем функции с @
                if func_name.startswith("@"):
                    logger.debug(
                        f"    Функция '{func_name}' игнорируется при проверке совместимости типов"
                    )
                    return

                # Получаем возвращаемый тип функции
                func_return_type = "unknown"

                # Проверяем в функциях
                if func_name in self.functions:
                    func_return_type = self.functions[func_name].get(
                        "return_type", "unknown"
                    )
                else:
                    # Ищем в scope'ах
                    for scope in self.all_scopes:
                        if (
                            scope.get("type") == "function"
                            and scope.get("function_name") == func_name
                        ):
                            func_return_type = scope.get("return_type", "unknown")
                            break

                if func_return_type != "unknown" and not self.are_types_compatible(
                    var_type, func_return_type
                ):
                    self.add_error(
                        f"переменной типа '{var_type}' присваивается результат функции '{func_name}' "
                        f"с возвращаемым типом '{func_return_type}'",
                        scope_idx,
                        node_idx,
                    )
                return  # Пропускаем дальнейшие проверки для вызова функции

        # Если это указатель
        if var_type.startswith("*"):
            pointed_type = var_type[1:]  # Тип, на который указывает указатель

            # Если берем адрес переменной (&x)
            if value.strip().startswith("&"):
                pointed_var = value.strip()[1:].strip()
                pointed_var_info = self.get_symbol_info(pointed_var, level)

                if pointed_var_info:
                    pointed_var_type = pointed_var_info.get("type")
                    if pointed_var_type != pointed_type:
                        self.add_error(
                            f"указатель '*{pointed_type}' не может указывать на переменную '{pointed_var}' типа '{pointed_var_type}'",
                            scope_idx,
                            node_idx,
                        )
            # Проверяем другие значения для указателей
            elif value.strip() != "null":  # null - допустимо для любого указателя
                value_type = self.guess_type_from_value(value)
                self.add_warning(
                    f"присвоение значения типа '{value_type}' указателю типа '*{pointed_type}'",
                    scope_idx,
                    node_idx,
                )
        else:
            # Проверяем обычные типы
            if var_type == "int":
                if value.startswith('"') or value.startswith("'"):
                    self.add_error(
                        f"нельзя присвоить строку переменной типа int",
                        scope_idx,
                        node_idx,
                    )
                elif value in ["True", "False"]:
                    self.add_error(
                        f"нельзя присвоить bool переменной типа int",
                        scope_idx,
                        node_idx,
                    )
            elif var_type == "str":
                if value.isdigit():
                    self.add_warning(
                        f"присвоение числа строковой переменной", scope_idx, node_idx
                    )

    # Type validation helper:
    def validate_type_operations(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует операции с типами"""
        if "expression_ast" in node:
            ast = node["expression_ast"]
            self.validate_ast_types(ast, node_idx, scope_idx, level)

    def validate_ast_types(self, ast: Dict, node_idx: int, scope_idx: int, level: int):
        """Рекурсивно валидирует типы в AST"""
        if not isinstance(ast, Mapping):
            return

        node_type = ast.get("type")

        if node_type == "binary_operation":
            # Проверяем совместимость типов операндов
            left_type = self.get_type_from_ast(
                ast.get("left"), scope_idx, node_idx, level
            )
            right_type = self.get_type_from_ast(
                ast.get("right"), scope_idx, node_idx, level
            )
            operator = ast.get("operator_symbol", "")

            if not self.can_operate_between_types(left_type, right_type, operator):
                self.add_error(
                    f"нельзя выполнить операцию '{operator}' "
                    f"между типами '{left_type}' и '{right_type}'",
                    scope_idx,
                    node_idx,
                )

            # Рекурсивно проверяем дочерние узлы
            self.validate_ast_types(ast.get("left"), node_idx, scope_idx, level)
            self.validate_ast_types(ast.get("right"), node_idx, scope_idx, level)

        elif node_type == "unary_operation":
            operand_type = self.get_type_from_ast(
                ast.get("operand"), scope_idx, node_idx, level
            )
            operator = ast.get("operator_symbol", "")

            if operator == "not" and operand_type != "bool":
                self.add_error(
                    f"оператор 'not' применяется к типу '{operand_type}', а не к bool",
                    scope_idx,
                    node_idx,
                )

            # Рекурсивно проверяем операнд
            self.validate_ast_types(ast.get("operand"), node_idx, scope_idx, level)

        elif node_type == "function_call":
            # Проверяем аргументы
            for arg in ast.get("arguments", []):
                self.validate_ast_types(arg, node_idx, scope_idx, level)

    def can_operate_between_types(self, type1: str, type2: str, operator: str) -> bool:
        """Проверяет, можно ли выполнить операцию между двумя типами"""
        # Если один из типов unknown, пропускаем проверку
        if type1 == "unknown" or type2 == "unknown":
            return True

        # Арифметические операции требуют числовых типов
        arithmetic_ops = [
            "+",
            "-",
            "*",
            "/",
            "//",
            "%",
            "**",
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "**=",
        ]

        if operator in arithmetic_ops:
            # ``+`` also implements string concatenation, but mixed
            # string/numeric arithmetic remains a type error.
            if operator == "+" and type1 == "str" and type2 == "str":
                return True

            # Все встроенные целочисленные и floating-point типы могут
            # участвовать в арифметике.
            numeric_types = [
                "int", "float", "double", "float16", "float32", "float64",
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64",
            ]
            return type1 in numeric_types and type2 in numeric_types

        # Операции сравнения
        comparison_ops = ["<", ">", "<=", ">=", "==", "!="]
        if operator in comparison_ops:
            # Можно сравнивать числовые типы между собой
            numeric_types = {
                "int", "float", "double", "float16", "float32", "float64",
                "int8", "int16", "int32", "int64",
                "uint8", "uint16", "uint32", "uint64",
            }
            if type1 in numeric_types and type2 in numeric_types:
                return True
            # Можно сравнивать строки со строками
            if type1 == "str" and type2 == "str":
                return True
            # Можно сравнивать булевы с булевыми
            if type1 == "bool" and type2 == "bool":
                return True
            return False

        # Логические операции
        logical_ops = ["and", "or"]
        if operator in logical_ops:
            return type1 == "bool" and type2 == "bool"

        return True

    def check_unused_variables(self, scope: Dict, scope_idx: int):
        """Warn about locals unused across the complete nested graph."""
        local_vars = scope.get("local_variables", [])
        graph = scope.get("graph", [])

        if not local_vars:
            return

        used_vars = set()

        def visit(node: Dict) -> None:
            if not isinstance(node, Mapping):
                return
            node_type = node.get("node", "")

            for dependency in node.get("dependencies", []) or []:
                if isinstance(dependency, str) and re.match(r"^[A-Za-z_]\w*$", dependency):
                    used_vars.add(dependency)

            # An assignment target is also a meaningful use for the current
            # language model (compound assignments and indexed writes rely on
            # this), but a declaration only introduces its target.
            if node_type != "declaration":
                for symbol in node.get("symbols", []) or []:
                    if isinstance(symbol, str) and re.match(r"^[A-Za-z_]\w*$", symbol):
                        used_vars.add(symbol)

            for key, value in node.items():
                if key in {"node", "content", "dependencies", "symbols", "source_line"}:
                    continue
                if isinstance(value, Mapping):
                    if value.get("type"):
                        self._collect_vars_from_ast(value, used_vars)
                    elif key in {"body", "else_block"}:
                        if isinstance(value.get("body"), list):
                            for child in value["body"]:
                                visit(child)
                elif isinstance(value, list):
                    for child in value:
                        if isinstance(child, Mapping):
                            if child.get("node"):
                                visit(child)
                            elif child.get("type"):
                                self._collect_vars_from_ast(child, used_vars)

        for node in graph:
            visit(node)

        # Находим неиспользуемые переменные
        for var in local_vars:
            if var == "_":
                continue
            # Пропускаем параметр 'self' в методах
            if var == "self" and scope.get("type") in ["constructor", "class_method"]:
                continue

            if var not in used_vars:
                self.add_warning(
                    f"переменная '{var}' объявлена, но нигде не используется",
                    scope_idx,
                    None,
                )

    def validate_return_paths(self, scope: Dict, scope_idx: int):
        """Проверяет, что все пути выполнения функции возвращают значение"""
        if scope.get("type") != "function":
            return

        return_type = scope.get("return_type", "None")
        if return_type == "None":
            return  # Функция void - не проверяем

        graph = scope.get("graph", [])
        has_return = False

        # Рекурсивно проверяем все узлы
        def check_node_for_return(node: Dict) -> bool:
            node_type = node.get("node")

            if node_type == "return":
                return True

            elif node_type == "if_statement":
                # Проверяем тело if
                if_body_has_return = False
                for body_node in node.get("body", []):
                    if check_node_for_return(body_node):
                        if_body_has_return = True
                        break

                # Проверяем elif блоки
                elif_has_return = False
                for elif_block in node.get("elif_blocks", []):
                    for body_node in elif_block.get("body", []):
                        if check_node_for_return(body_node):
                            elif_has_return = True
                            break
                    if elif_has_return:
                        break

                # Проверяем else блок
                else_has_return = False
                else_block = node.get("else_block")
                if else_block:
                    for body_node in else_block.get("body", []):
                        if check_node_for_return(body_node):
                            else_has_return = True
                            break

                # Если есть else, проверяем, что все пути возвращают значение
                if else_block:
                    return if_body_has_return and elif_has_return and else_has_return
                else:
                    # Если нет else, функция может не возвращать значение
                    return False

            elif node_type in ["while_loop", "for_loop"]:
                # Циклы не гарантируют возврат
                return False

            return False

        # Проверяем все узлы в графе
        for node in graph:
            if check_node_for_return(node):
                has_return = True
                break

        if not has_return:
            self.add_warning(
                f"функция объявлена как возвращающая '{return_type}', "
                f"но не все пути выполнения возвращают значение",
                scope_idx,
                None,
            )

    def check_division_by_zero(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Проверяет деление на ноль"""
        if node.get("node") in ["assignment", "declaration"]:
            content = node.get("content", "")

            # Ищем операции деления
            if "/" in content or "//" in content or "/=" in content:
                # Упрощенная проверка
                if "/ 0" in content or "// 0" in content:
                    self.add_warning("возможное деление на ноль", scope_idx, node_idx)

                # Более сложная проверка для переменных
                pattern = (
                    r"[/](?:\s*0\b|\s*[a-zA-Z_][a-zA-Z0-9_]*(?:\s*[*+-/]\s*\w+)*\s*)"
                )
                if re.search(pattern, content):
                    # Проверяем, может ли переменная быть нулем
                    self.add_warning(
                        "возможное деление на переменную, которая может быть нулем",
                        scope_idx,
                        node_idx,
                    )

    def check_loop_conditions(self, scope: Dict, scope_idx: int):
        """Проверяет условия циклов на потенциальные проблемы"""
        graph = scope.get("graph", [])

        for node_idx, node in enumerate(graph):
            if node.get("node") == "while_loop":
                condition = node.get("condition", {})

                # Проверяем вечные циклы (while True)
                if condition.get("value") == "True":
                    self.add_warning("бесконечный цикл while True", scope_idx, node_idx)

                # Проверяем невозможные условия (while False)
                if condition.get("value") == "False":
                    self.add_warning(
                        "цикл while с условием, которое всегда ложно",
                        scope_idx,
                        node_idx,
                    )

            elif node.get("node") == "for_loop":
                iterable = node.get("iterable", {})

                # Проверяем пустые диапазоны range()
                if iterable.get("type") == "RANGE_CALL":
                    args = iterable.get("arguments", {})

                    # range(x, x) - пустой диапазон
                    if args.get("start") == args.get("stop"):
                        self.add_warning(
                            "цикл for с пустым диапазоном range()", scope_idx, node_idx
                        )

                    # range(x, y) где x > y без отрицательного шага
                    if (
                        args.get("start")
                        and args.get("stop")
                        and args.get("step") not in ["-1", "-2"]
                    ):
                        # Упрощенная проверка
                        try:
                            start = int(args.get("start"))
                            stop = int(args.get("stop"))
                            if start > stop:
                                self.add_warning(
                                    "цикл for с start > stop без отрицательного шага",
                                    scope_idx,
                                    node_idx,
                                )
                        except Exception:
                            pass

    def check_memory_leaks(self, scope: Dict, scope_idx: int):
        """Проверяет потенциальные утечки памяти с указателями"""
        graph = scope.get("graph", [])
        level = scope.get("level", 0)

        pointer_declarations = {}  # {pointer_name: node_idx}
        pointer_deletes = set()  # pointer_names that were deleted

        for node_idx, node in enumerate(graph):
            node_type = node.get("node")

            # Отслеживаем объявления указателей
            if node_type == "declaration":
                operations = node.get("operations", [])
                for op in operations:
                    if op.get("type") == "NEW_POINTER":
                        symbols = node.get("symbols", [])
                        if symbols:
                            pointer_declarations[symbols[0]] = node_idx

            # Отслеживаем удаление указателей
            elif node_type in ["delete"]:
                symbols = node.get("symbols", [])
                for symbol in symbols:
                    pointer_deletes.add(symbol)

        # Проверяем объявленные, но не удаленные указатели
        for pointer_name, decl_idx in pointer_declarations.items():
            if pointer_name not in pointer_deletes:
                # Проверяем, что указатель не был удален в родительском scope
                # или что это не возвращаемое значение
                self.add_warning(
                    f"указатель '{pointer_name}' объявлен, но не удален (возможная утечка памяти)",
                    scope_idx,
                    decl_idx,
                )

    def get_scope_by_level(self, level: int) -> Optional[Dict]:
        """Находит scope по уровню"""
        for scope in self.all_scopes:
            if scope.get("level") == level:
                return scope
        return None

    def guess_type_from_value(self, value) -> str:
        """Пытается определить тип по значению"""
        # Если value - строка
        if isinstance(value, str):
            value = value.strip()

            # Проверяем арифметические выражения
            if any(op in value for op in ["+", "-", "*", "/"]):
                # Простая эвристика: если есть цифры - предположительно int
                if any(c.isdigit() for c in value):
                    return "int"
                return "unknown"

            # Остальные проверки как раньше
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                return "str"

            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return "int"

            if re.match(r"^-?\d+\.\d+$", value):
                return "float"

            if value in ["True", "False"]:
                return "bool"

            if value == "None":
                return "None"

            if value == "null":
                return "null"

            if value.startswith("&"):
                return "pointer"

            if value.startswith("*"):
                return "dereference"

            if value.startswith("["):
                return "list"

            if value.startswith("{"):
                if ":" in value:
                    return "dict"
                else:
                    return "set"

            # Если это вызов функции (содержит скобки)
            if "(" in value and ")" in value:
                # Извлекаем имя функции
                func_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\(", value)
                if func_match:
                    func_name = func_match.group(1)
                    # Проверяем тип возвращаемого значения функции
                    if func_name in self.functions:
                        func_info = self.functions[func_name]
                        return func_info.get("return_type", "unknown")
                    elif func_name in self.builtin_functions:
                        return self.builtin_functions[func_name]["return_type"]

            return "unknown"

        # Если value - AST (словарь)
        elif isinstance(value, Mapping):
            return self.get_type_from_ast(value, None, None, None)

        return "unknown"

    def are_types_compatible(self, target_type: str, value_type: str) -> bool:
        """Проверяет совместимость типов"""
        target_type = (target_type or "").strip()
        value_type = (value_type or "").strip()

        # Если типы равны - совместимы
        if target_type == value_type:
            return True

        # Raw pointers follow the C void* interoperability rule. Ownership
        # and unsafe-boundary checks are performed separately; this rule only
        # answers whether the raw pointer representations are compatible.
        if target_type.startswith("*") and value_type.startswith("*"):
            target_base = target_type[1:].strip()
            value_base = value_type[1:].strip()

            return (
                target_base == value_base
                or target_base == "void"
                or value_base == "void"
            )

        # A borrow declaration binds a view to an existing compatible owner.
        # The lifetime pass separately verifies that the source stays alive and
        # that mutable/immutable aliases do not conflict.
        if target_type.startswith("&mut ") and target_type[5:].strip() == value_type:
            return True
        if target_type.startswith("&") and not target_type.startswith("&mut "):
            if target_type[1:].strip() == value_type:
                return True

        # Null совместим с любым указателем
        if value_type == "null" and target_type.startswith("*"):
            return True

        # None совместим с любым типом, если target_type - None
        if value_type == "None" and target_type == "None":
            return True

        # Container literals and factory methods carry their element type in
        # the declaration, while the expression AST only knows the container
        # kind at this stage.
        generic_containers = ("list[", "dict[", "tuple[", "array[")
        if target_type.startswith(generic_containers) and value_type in {
            "list", "dict", "tuple", "array"
        }:
            if value_type == "list":
                return target_type.startswith(("list[", "array["))
            return target_type.startswith(f"{value_type}[")

        # Ошибка: если функция должна возвращать int, а возвращает str
        if target_type == "int" and value_type == "str":
            return False

        if target_type == "str" and value_type == "int":
            return False

        # Numeric literals are widened to the declared numeric type.
        float_types = {"float", "double", "float16", "float32", "float64"}
        integer_types = {
            "int", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "size_t",
        }
        if target_type in KNOWN_C_TYPES and value_type in integer_types:
            return True
        if target_type in float_types and value_type in float_types | {"int"}:
            return True
        if target_type in integer_types and value_type in integer_types:
            return True

        # Упрощенные правила совместимости
        compatibility_rules = {
            "int": ["bool"],  # int может принимать bool (True=1, False=0)
            "bool": ["int"],  # bool может принимать int (0=False, не 0=True)
        }

        if (
            target_type in compatibility_rules
            and value_type in compatibility_rules[target_type]
        ):
            return True

        # Если value_type - конкретный тип, а target_type - указатель на тот же тип
        if target_type.startswith("*") and f"*{value_type}" == target_type:
            return True

        return False

    def find_symbol_in_scope(self, symbol_name: str, current_level: int) -> bool:
        """Ищет символ в текущем или родительских scope'ах"""
        # Проверяем встроенные функции
        if symbol_name in self.builtin_functions:
            return True

        # Проверяем пользовательские функции
        if symbol_name in self.functions:
            return True

        # Проверяем классы
        if symbol_name in self.classes:
            return True

        current_scope = self._active_scope
        if not current_scope or current_scope.get("level") != current_level:
            current_scope = next(
                (scope for scope in reversed(self.all_scopes)
                 if scope.get("level") == current_level),
                None,
            )

        visited = set()
        while current_scope is not None and id(current_scope) not in visited:
            visited.add(id(current_scope))
            if symbol_name in (current_scope.get("symbol_table") or {}):
                return True
            parent_level = current_scope.get("parent_scope")
            current_scope = next(
                (scope for scope in reversed(self.all_scopes)
                 if scope.get("level") == parent_level),
                None,
            ) if parent_level is not None else None

        return False

    def get_symbol_info(self, symbol_name: str, current_level: int) -> Optional[Dict]:
        """Получает информацию о символе из текущего или родительских scope'ов"""
        # Сначала проверяем функции
        if symbol_name in self.functions:
            return self.functions[symbol_name]

        # Проверяем встроенные функции
        if symbol_name in self.builtin_functions:
            return {"name": symbol_name, "type": "function", "key": "builtin_function"}

        # Проверяем классы
        if symbol_name in self.classes:
            return self.classes[symbol_name]

        current_scope = self._active_scope
        if not current_scope or current_scope.get("level") != current_level:
            current_scope = next(
                (scope for scope in reversed(self.all_scopes)
                 if scope.get("level") == current_level),
                None,
            )

        visited = set()
        while current_scope is not None and id(current_scope) not in visited:
            visited.add(id(current_scope))
            symbol_table = current_scope.get("symbol_table") or {}
            if symbol_name in symbol_table:
                return symbol_table[symbol_name]
            parent_level = current_scope.get("parent_scope")
            current_scope = next(
                (scope for scope in reversed(self.all_scopes)
                 if scope.get("level") == parent_level),
                None,
            ) if parent_level is not None else None

        return None

    def validate_method_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов метода объекта"""
        obj_name = node.get("object")
        method_name = node.get("method")
        arguments = node.get("arguments", [])

        # Проверяем объект
        if obj_name and not self.find_symbol_in_scope(obj_name, level):
            self.add_error(f"объект '{obj_name}' не объявлен", scope_idx, node_idx)
        elif obj_name and self.is_variable_deleted(obj_name, level):
            self.add_error(f"объект '{obj_name}' был удален", scope_idx, node_idx)

        # Проверяем аргументы метода
        for arg in arguments:
            self._validate_argument(
                arg, f"{obj_name}.{method_name}", scope_idx, node_idx, level
            )

    def validate_static_method_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов статического метода"""
        class_name = node.get("class_name")
        method_name = node.get("method")
        arguments = node.get("arguments", [])

        # Проверяем, существует ли класс
        class_symbol = self._find_class_symbol(class_name, level)
        if not class_symbol:
            self.add_error(f"класс '{class_name}' не объявлен", scope_idx, node_idx)

        # Проверяем аргументы
        for arg in arguments:
            self._validate_argument(
                arg, f"{class_name}.{method_name}", scope_idx, node_idx, level
            )

    def _find_class_symbol(self, class_name: str, level: int) -> Optional[Dict]:
        """Находит символ класса в таблице символов"""
        for scope_info in self.scopes_info:
            if scope_info["level"] <= level:
                symbol_table = scope_info.get("symbol_table", {})
                class_symbol = symbol_table.get(class_name)
                if class_symbol and class_symbol.get("key") == "class":
                    return class_symbol
        return None

    def validate_builtin_function_call(
        self, node: Dict, node_idx: int, scope_idx: int, symbol_table: Dict, level: int
    ):
        """Валидирует вызов встроенной функции"""
        func_name = node.get("function")
        arguments = node.get("arguments", [])

        # Проверяем аргументы встроенных функций
        for arg in arguments:
            self._validate_argument(arg, func_name, scope_idx, node_idx, level)

    def get_report(self) -> Dict:
        """Возвращает отчет о проверке"""
        return self.get_diagnostic_report().as_dict()

    def get_diagnostic_report(self) -> DiagnosticReport:
        """Return the canonical typed report without compatibility projections."""
        return DiagnosticReport(tuple(self.diagnostics))

    def validate_inheritance_hierarchy(self, scope: Dict, scope_idx: int):
        """Проверяет корректность иерархии наследования классов"""
        if scope.get("type") != "class_declaration":
            return

        class_name = scope.get("class_name")
        base_classes = scope.get("base_classes", [])

        if not base_classes:
            return

        # 1. Проверяем циклические зависимости
        for base_class in base_classes:
            # Находим базовый класс в all_scopes
            base_scope = None
            for s in self.all_scopes:
                if (
                    s.get("type") == "class_declaration"
                    and s.get("class_name") == base_class
                ):
                    base_scope = s
                    break

            if base_scope:
                # Проверяем, не наследует ли базовый класс от текущего (цикл)
                if class_name in base_scope.get("base_classes", []):
                    self.add_error(
                        f"циклическое наследование: класс '{class_name}' и '{base_class}' наследуют друг от друга",
                        scope_idx,
                        None,
                    )

        # 2. Проверяем дублирование методов в MRO
        all_methods = []
        method_sources = {}  # {method_name: [class_name, ...]}

        # Собираем методы текущего класса
        for method in scope.get("methods", []):
            method_name = method.get("name")
            if method_name not in method_sources:
                method_sources[method_name] = []
            method_sources[method_name].append(class_name)
            all_methods.append(method_name)

        # Рекурсивно собираем методы из базовых классов
        def collect_base_methods(base_class_name, visited=None):
            if visited is None:
                visited = set()

            if base_class_name in visited:
                return
            visited.add(base_class_name)

            # Находим базовый класс
            base_scope = None
            for s in self.all_scopes:
                if (
                    s.get("type") == "class_declaration"
                    and s.get("class_name") == base_class_name
                ):
                    base_scope = s
                    break

            if not base_scope:
                return

            # Собираем методы базового класса
            for method in base_scope.get("methods", []):
                method_name = method.get("name")
                if method_name not in method_sources:
                    method_sources[method_name] = []
                method_sources[method_name].append(base_class_name)
                all_methods.append(method_name)

            # Рекурсивно для родительских классов
            for parent in base_scope.get("base_classes", []):
                collect_base_methods(parent, visited)

        for base_class in base_classes:
            collect_base_methods(base_class)

        # Проверяем конфликты методов (одинаковые имена в разных классах)
        for method_name, sources in method_sources.items():
            if len(sources) > 1:
                # Метод есть в нескольких классах - проверяем, переопределен ли он
                if class_name in sources:
                    # Текущий класс переопределяет метод
                    self.add_warning(
                        f"метод '{method_name}' переопределен в классе '{class_name}' "
                        f"(также определен в: {', '.join([c for c in sources if c != class_name])})",
                        scope_idx,
                        None,
                    )
                else:
                    # Конфликт в базовых классах
                    self.add_error(
                        f"конфликт методов: '{method_name}' определен в нескольких базовых классах "
                        f"({', '.join(sources)}) без переопределения",
                        scope_idx,
                        None,
                    )

    def check_method_resolution_order(self, class_name: str):
        """Проверяет порядок разрешения методов (MRO)"""
        # Находим класс
        class_scope = None
        for scope in self.all_scopes:
            if (
                scope.get("type") == "class_declaration"
                and scope.get("class_name") == class_name
            ):
                class_scope = scope
                break

        if not class_scope:
            return

        base_classes = class_scope.get("base_classes", [])
        if not base_classes:
            return

        # Простой алгоритм MRO (C3 linearization)
        def compute_mro(cls_name, visited=None):
            if visited is None:
                visited = set()

            if cls_name in visited:
                return []
            visited.add(cls_name)

            # Находим класс
            cls_scope = None
            for s in self.all_scopes:
                if (
                    s.get("type") == "class_declaration"
                    and s.get("class_name") == cls_name
                ):
                    cls_scope = s
                    break

            if not cls_scope:
                return [cls_name]

            result = [cls_name]
            for base in cls_scope.get("base_classes", []):
                result.extend(compute_mro(base, visited))

            return result

        try:
            mro = compute_mro(class_name)
            # Проверяем дубликаты в MRO (циклическое наследование)
            if len(mro) != len(set(mro)):
                self.add_error(
                    f"циклическое наследование в MRO класса '{class_name}'",
                    self.all_scopes.index(class_scope)
                    if class_scope in self.all_scopes
                    else None,
                    None,
                )

            return mro
        except RecursionError:
            self.add_error(
                f"бесконечная рекурсия в наследовании класса '{class_name}'",
                self.all_scopes.index(class_scope)
                if class_scope in self.all_scopes
                else None,
                None,
            )
            return []

    def validate_thread_functions(self, scope: Dict, scope_idx: int):
        """Проверяет функции для использования в потоках"""
        if scope.get("type") != "function":
            return

        func_name = scope.get("function_name")
        if not func_name:
            return

        # Проверяем, используется ли функция как callback для потока
        for other_scope in self.all_scopes:
            for node_idx, node in enumerate(other_scope.get("graph", [])):
                if (
                    node.get("node") == "c_call"
                    and node.get("function") == "pthread_create"
                ):
                    args = node.get("arguments", [])
                    if len(args) >= 3 and args[2] == func_name:
                        # Эта функция передается в pthread_create

                        # 1. Проверяем сигнатуру функции
                        parameters = scope.get("parameters", [])
                        if len(parameters) != 1:
                            self.add_error(
                                f"функция '{func_name}' передается в pthread_create, "
                                f"но должна принимать ровно 1 параметр (void*), а принимает {len(parameters)}",
                                scope_idx,
                                None,
                            )
                        else:
                            param_type = parameters[0].get("type", "")
                            if param_type not in ["None", "void*", "*void"]:
                                self.add_warning(
                                    f"функция '{func_name}' передается в pthread_create, "
                                    f"параметр должен быть void* (получен: {param_type})",
                                    scope_idx,
                                    None,
                                )

                        # 2. Проверяем тип возврата
                        return_type = scope.get("return_type", "")
                        if return_type not in ["None", "void*", "*void"]:
                            self.add_warning(
                                f"функция потока '{func_name}' должна возвращать void* (возвращает: {return_type})",
                                scope_idx,
                                None,
                            )

    def check_unused_parameters(self, scope: Dict, scope_idx: int):
        """Находит неиспользуемые параметры функций и методов"""
        scope_type = scope.get("type")

        if scope_type not in ["function", "constructor", "class_method"]:
            return

        parameters = scope.get("parameters", [])
        if not parameters:
            return

        # Собираем все используемые переменные в теле функции
        used_vars = set()
        graph = scope.get("graph", [])

        def visit(node: Dict) -> None:
            if not isinstance(node, Mapping):
                return
            for dep in node.get("dependencies", []) or []:
                if isinstance(dep, str) and re.match(r"^[A-Za-z_]\w*$", dep):
                    used_vars.add(dep)
            for symbol in node.get("symbols", []) or []:
                if isinstance(symbol, str) and re.match(r"^[A-Za-z_]\w*$", symbol):
                    used_vars.add(symbol)
            for value in node.values():
                if isinstance(value, Mapping):
                    if value.get("node"):
                        visit(value)
                    elif value.get("type"):
                        self._collect_vars_from_ast(value, used_vars)
                elif isinstance(value, list):
                    for child in value:
                        if isinstance(child, Mapping):
                            if child.get("node"):
                                visit(child)
                            elif child.get("type"):
                                self._collect_vars_from_ast(child, used_vars)

        for node in graph:
            visit(node)

        # Проверяем каждый параметр
        for param in parameters:
            param_name = param.get("name")
            if not param_name:
                continue

            if param_name == "_":
                continue

            # Пропускаем self в методах
            if scope_type in ["constructor", "class_method"] and param_name == "self":
                continue

            if param_name not in used_vars:
                # Для конструкторов это ошибка, для других методов - предупреждение
                if scope.get("method_name") == "__init__":
                    self.add_error(
                        f"параметр конструктора '{param_name}' не используется",
                        scope_idx,
                        None,
                    )
                else:
                    self.add_warning(
                        f"параметр '{param_name}' не используется", scope_idx, None
                    )

    def validate_pointer_usage(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Проверяет корректное использование указателей"""
        node_type = node.get("node")

        if node_type == "declaration":
            # Проверяем объявление указателей
            var_type = node.get("var_type", "")
            if var_type.startswith("*"):
                # Это указатель - проверяем инициализацию
                operations = node.get("operations", [])
                for op in operations:
                    if op.get("type") == "GET_ADDRESS":
                        pointed_var = op.get("of", "")
                        if pointed_var:
                            # Проверяем, что переменная существует
                            if not self.find_symbol_in_scope(pointed_var, level):
                                self.add_error(
                                    f"указатель создается для несуществующей переменной '{pointed_var}'",
                                    scope_idx,
                                    node_idx,
                                )

        elif node_type == "function_call" or node_type == "c_call":
            # Проверяем передачу указателей в функции
            func_name = node.get("function", "")
            arguments = node.get("arguments", [])

            # Проверяем pthread_create
            if func_name == "pthread_create":
                if len(arguments) >= 4:
                    thread_data_arg = arguments[3]
                    # Проверяем, что 4-й аргумент - указатель
                    if isinstance(thread_data_arg, str) and thread_data_arg.isalpha():
                        var_info = self.get_symbol_info(thread_data_arg, level)
                        if var_info:
                            var_type = var_info.get("type", "")
                            if not var_type.startswith("*"):
                                self.add_warning(
                                    f"в pthread_create передается не указатель '{thread_data_arg}' типа '{var_type}'",
                                    scope_idx,
                                    node_idx,
                                )

            # Проверяем pthread_join
            elif func_name == "pthread_join":
                if len(arguments) >= 1:
                    thread_arg = arguments[0]
                    if isinstance(thread_arg, str) and thread_arg.isalpha():
                        var_info = self.get_symbol_info(thread_arg, level)
                        if not var_info:
                            self.add_error(
                                f"переменная потока '{thread_arg}' не объявлена",
                                scope_idx,
                                node_idx,
                            )

        elif node_type == "dereference_read" or node_type == "dereference_write":
            # Проверяем разыменование указателей
            symbols = node.get("symbols", [])
            for symbol in symbols:
                var_info = self.get_symbol_info(symbol, level)
                if var_info:
                    var_type = var_info.get("type", "")
                    if not var_type.startswith("*"):
                        self.add_error(
                            f"попытка разыменования не указателя '{symbol}' типа '{var_type}'",
                            scope_idx,
                            node_idx,
                        )

    def validate_array_bounds(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Проверяет выход за границы массивов/списков"""
        node_type = node.get("node")

        if node_type == "index_access":
            # Чтение по индексу
            variable = node.get("variable", "")
            index = node.get("index", {})

            if variable and index:
                # Получаем информацию о списке/массиве
                var_info = self.get_symbol_info(variable, level)
                if var_info:
                    var_type = var_info.get("type", "")
                    if "list" in var_type or "array" in var_type:
                        # Пытаемся получить статическое значение индекса
                        index_value = self._get_static_value_from_ast(index, level)
                        if index_value is not None:
                            # Проверяем отрицательные индексы
                            if index_value < 0:
                                self.add_warning(
                                    f"использование отрицательного индекса {index_value} для '{variable}'",
                                    scope_idx,
                                    node_idx,
                                )

        elif node_type == "index_assignment":
            # Присваивание по индексу
            try:
                variable = node.get("variable", "")
                index = node.get("index", {})

                if variable and index:
                    var_info = self.get_symbol_info(variable, level)
                    if var_info:
                        var_type = var_info.get("type", "")
                        if "list" in var_type or "array" in var_type:
                            index_value = self._get_static_value_from_ast(index, level)
                            if index_value is not None and index_value < 0:
                                self.add_warning(
                                    f"присваивание по отрицательному индексу {index_value} для '{variable}'",
                                    scope_idx,
                                    node_idx,
                                )
            except Exception as e:
                logger.error(e)

        elif node_type == "slice_access" or node_type == "slice_assignment":
            # Работа со срезами
            variable = node.get("variable", "")
            start = node.get("start")
            stop = node.get("stop")

            if variable:
                # Проверяем отрицательные индексы в срезах
                if start:
                    start_value = self._get_static_value_from_ast(start, level)
                    if start_value is not None and start_value < 0:
                        self.add_warning(
                            f"использование отрицательного начала среза {start_value} для '{variable}'",
                            scope_idx,
                            node_idx,
                        )

                if stop:
                    stop_value = self._get_static_value_from_ast(stop, level)
                    if stop_value is not None and stop_value < 0:
                        self.add_warning(
                            f"использование отрицательного конца среза {stop_value} для '{variable}'",
                            scope_idx,
                            node_idx,
                        )

    def validate_string_operations(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Проверяет операции со строками"""
        node_type = node.get("node")

        if node_type == "method_call":
            obj_name = node.get("object", "")
            method_name = node.get("method", "")

            if obj_name and method_name:
                obj_info = self.get_symbol_info(obj_name, level)
                if obj_info:
                    obj_type = obj_info.get("type", "")

                    if obj_type == "str":
                        if method_name == "upper":
                            # Проверяем, используется ли результат
                            parent_node = self._find_parent_node(scope_idx, node_idx)
                            if parent_node and parent_node.get("node") not in [
                                "assignment",
                                "declaration",
                            ]:
                                self.add_warning(
                                    f"результат метода '{obj_name}.upper()' не сохраняется",
                                    scope_idx,
                                    node_idx,
                                )

                        elif method_name == "split":
                            arguments = node.get("arguments", [])
                            if len(arguments) == 1:
                                arg = arguments[0]
                                if (
                                    isinstance(arg, Mapping)
                                    and arg.get("type") == "literal"
                                ):
                                    value = arg.get("value", "")
                                    if value == " ":
                                        # Правильный разделитель
                                        pass
                                    elif value == "":
                                        self.add_warning(
                                            f"пустой разделитель в '{obj_name}.split(\"\")' может привести к неожиданным результатам",
                                            scope_idx,
                                            node_idx,
                                        )
                                    else:
                                        self.add_warning(
                                            f"использование нестандартного разделителя '{value}' в split",
                                            scope_idx,
                                            node_idx,
                                        )

    def validate_c_function_calls(
        self, node: Dict, node_idx: int, scope_idx: int, level: int
    ):
        """Проверяет вызовы C-функций (начинающиеся с @)"""
        node_type = node.get("node")

        if node_type == "c_call":
            func_name = node.get("function", "")
            arguments = node.get("arguments", [])

            # Проверяем известные C-функции
            if func_name in ["pthread_create", "pthread_join"]:
                # Проверяем количество аргументов
                expected_args = 4 if func_name == "pthread_create" else 2
                if len(arguments) != expected_args:
                    self.add_error(
                        f"функция '{func_name}' ожидает {expected_args} аргументов, получено {len(arguments)}",
                        scope_idx,
                        node_idx,
                    )

                # Проверяем типы аргументов
                if func_name == "pthread_create":
                    # 1-й аргумент: &thread (адрес переменной pthread_t)
                    if len(arguments) > 0:
                        arg = arguments[0]
                        if isinstance(arg, str) and arg.startswith("&"):
                            var_name = arg[1:].strip()
                            var_info = self.get_symbol_info(var_name, level)
                            if not var_info:
                                self.add_error(
                                    f"переменная '{var_name}' для &thread не объявлена",
                                    scope_idx,
                                    node_idx,
                                )

                    # 2-й аргумент: NULL
                    if len(arguments) > 1:
                        arg = arguments[1]
                        if arg != "NULL" and arg != "nullptr":
                            self.add_warning(
                                f"второй аргумент pthread_create должен быть NULL (получен: {arg})",
                                scope_idx,
                                node_idx,
                            )

                    # 3-й аргумент: функция
                    if len(arguments) > 2:
                        func_arg = arguments[2]
                        if isinstance(func_arg, str) and func_arg.isalpha():
                            # Проверяем, что функция существует
                            func_found = False
                            for scope in self.all_scopes:
                                if (
                                    scope.get("type") == "function"
                                    and scope.get("function_name") == func_arg
                                ):
                                    func_found = True
                                    break

                            if not func_found:
                                self.add_error(
                                    f"функция '{func_arg}' для потока не объявлена",
                                    scope_idx,
                                    node_idx,
                                )

    def check_missing_declarations(self, scope: Dict, scope_idx: int):
        """Проверяет отсутствующие объявления"""
        # Проверяем C-типы
        for node_idx, node in enumerate(scope.get("graph", [])):
            if node.get("node") == "declaration":
                var_type = node.get("var_type", "")
                if var_type in ["pthread_t"]:
                    # Проверяем, импортирован ли соответствующий заголовок
                    has_pthread_import = False
                    for module_scope in self.all_scopes:
                        if module_scope.get("level") == 0:
                            for module_node in module_scope.get("graph", []):
                                if module_node.get("node") == "c_import":
                                    header = module_node.get("header", "")
                                    if "pthread" in header.lower():
                                        has_pthread_import = True
                                        break

                    if not has_pthread_import:
                        self.add_warning(
                            f"используется тип '{var_type}' без импорта pthread.h",
                            scope_idx,
                            node_idx,
                        )

    # Вспомогательные методы

    def _collect_vars_from_ast(self, ast: Dict, used_vars: set):
        """Collect variable references from any expression AST variant."""
        if not isinstance(ast, Mapping):
            return

        node_type = ast.get("type")

        if node_type == "variable":
            var_name = ast.get("value") or ast.get("name")
            if var_name and re.match(r"^[A-Za-z_]\w*$", var_name):
                used_vars.add(var_name)
            return

        if node_type == "RANGE_CALL":
            arguments = ast.get("arguments", {})
            if isinstance(arguments, Mapping):
                for value in arguments.values():
                    if isinstance(value, str) and re.match(r"^[A-Za-z_]\w*$", value):
                        used_vars.add(value)
            return

        if node_type in {"literal", "unknown", "empty"}:
            return

        if node_type in {"attribute_access", "complex_attribute_access", "method_call"}:
            obj_name = ast.get("object")
            if obj_name and re.match(r"^[A-Za-z_]\w*$", obj_name):
                used_vars.add(obj_name)

        ignored_keys = {
            "type", "name", "value", "attribute", "function", "method",
            "operator", "operator_symbol", "data_type", "original",
        }
        for key, child in ast.items():
            if key in ignored_keys:
                continue
            if isinstance(child, Mapping):
                self._collect_vars_from_ast(child, used_vars)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, Mapping):
                        self._collect_vars_from_ast(item, used_vars)

    def _get_static_value_from_ast(self, ast: Dict, level: int):
        """Пытается получить статическое значение из AST"""
        if not isinstance(ast, Mapping):
            return None

        node_type = ast.get("type")

        if node_type == "literal":
            return ast.get("value")

        elif node_type == "variable":
            var_name = ast.get("value")
            # Можем попытаться отследить константы
            var_info = self.get_symbol_info(var_name, level)
            if var_info and var_info.get("key") == "const":
                value = var_info.get("value")
                if isinstance(value, Mapping) and value.get("type") == "literal":
                    return value.get("value")

        return None

    def _find_parent_node(self, scope_idx: int, node_idx: int):
        """Находит родительский узел (если есть)"""
        if scope_idx >= len(self.all_scopes):
            return None

        graph = self.all_scopes[scope_idx].get("graph", [])
        # Простая реализация - возвращаем предыдущий узел
        if node_idx > 0 and node_idx - 1 < len(graph):
            return graph[node_idx - 1]
        return None

    def _validate_class_method_calls(self, scope: Dict, scope_idx: int):
        """Проверяет вызовы методов в контексте наследования"""
        class_name = scope.get("class_name", "")
        method_name = scope.get("method_name", "")

        # Для методов get_age проверяем конфликты с родительскими классами
        if method_name == "get_age":
            # Находим информацию о классе
            class_scope = None
            for s in self.all_scopes:
                if (
                    s.get("type") == "class_declaration"
                    and s.get("class_name") == class_name
                ):
                    class_scope = s
                    break

            if class_scope:
                base_classes = class_scope.get("base_classes", [])
                for base_class in base_classes:
                    # Проверяем, есть ли метод get_age в базовом классе
                    base_scope = None
                    for s in self.all_scopes:
                        if (
                            s.get("type") == "class_declaration"
                            and s.get("class_name") == base_class
                        ):
                            base_scope = s
                            break

                    if base_scope:
                        for method in base_scope.get("methods", []):
                            if method.get("name") == "get_age":
                                # Проверяем совместимость сигнатур
                                current_params = scope.get("parameters", [])
                                base_params = method.get("parameters", [])

                                if len(current_params) != len(base_params):
                                    self.add_warning(
                                        f"метод '{method_name}' переопределен с изменением количества параметров "
                                        f"(было {len(base_params)}, стало {len(current_params)})",
                                        scope_idx,
                                        None,
                                    )
                                break

    def _collect_function_metrics(self, scope: Dict, scope_idx: int):
        """Собирает метрики сложности кода"""
        if "graph" not in scope:
            return

        graph = scope.get("graph", [])

        # Считаем сложность (упрощенный цикломатический)
        complexity = 1  # базовая сложность
        condition_count = 0
        loop_count = 0

        for node in graph:
            node_type = node.get("node", "")
            if node_type in ["if_statement", "while_loop", "for_loop"]:
                condition_count += 1
            if node_type in ["while_loop", "for_loop"]:
                loop_count += 1

        complexity += condition_count + loop_count

        # Пороги предупреждений
        if complexity > 10:
            self.add_warning(
                f"высокая цикломатическая сложность функции ({complexity})",
                scope_idx,
                None,
            )

        if loop_count > 3:
            self.add_warning(
                f"слишком много циклов в функции ({loop_count})",
                scope_idx,
                None,
            )

        # Считаем длину функции (узлы)
        if len(graph) > 50:
            self.add_warning(
                f"функция слишком длинная ({len(graph)} операций)",
                scope_idx,
                None,
            )

    def check_undefined_methods(self, scope: Dict, scope_idx: int):
        """Проверяет, что все используемые методы определены в классе или его родителях"""
        scope_type = scope.get("type")

        if scope_type not in ["function", "constructor", "class_method", "module"]:
            return

        # Собираем все вызовы методов в текущем scope
        method_calls = []

        graph = scope.get("graph", [])
        for node_idx, node in enumerate(graph):
            node_type = node.get("node")

            if node_type == "method_call":
                obj_name = node.get("object", "")
                method_name = node.get("method", "")

                if obj_name and method_name:
                    # Добавляем в список для проверки
                    method_calls.append(
                        {
                            "obj": obj_name,
                            "method": method_name,
                            "node_idx": node_idx,
                            "content": node.get("content", ""),
                        }
                    )

            # Также проверяем вызовы методов в AST
            if "expression_ast" in node:
                self._extract_method_calls_from_ast(
                    node["expression_ast"],
                    method_calls,
                    node_idx,
                    node.get("content", ""),
                )

        # Уникальные проверки (чтобы избежать дублирования)
        checked_calls = set()

        # Проверяем каждый вызов метода
        for call in method_calls:
            obj_name = call["obj"]
            method_name = call["method"]
            node_idx = call["node_idx"]
            content = call["content"]

            # Пропускаем, если уже проверяли эту комбинацию
            call_key = f"{obj_name}.{method_name}.{node_idx}"
            if call_key in checked_calls:
                continue
            checked_calls.add(call_key)

            # Получаем информацию об объекте
            obj_info = self.get_symbol_info(obj_name, scope.get("level", 0))
            if not obj_info:
                # Объект не найден - другая проверка это поймает
                continue

            obj_type = obj_info.get("type", "")

            # Пропускаем, если тип неизвестен
            if not obj_type or obj_type == "unknown":
                continue

            # Проверяем, является ли тип классом
            class_found = False

            # Ищем класс во всех scope'ах
            for s in self.all_scopes:
                if (
                    s.get("type") == "class_declaration"
                    and s.get("class_name") == obj_type
                ):
                    class_found = True
                    # Добавляем в classes для будущих проверок
                    if obj_type not in self.classes:
                        self._add_class_to_registry(s)

                    # Проверяем метод
                    if not self._method_exists_in_class_hierarchy(
                        obj_type, method_name
                    ):
                        self.add_error(
                            f"метод '{method_name}' не определен для объекта типа '{obj_type}'",
                            scope_idx,
                            node_idx,
                        )
                    break

            if not class_found:
                # Проверяем встроенные типы
                if obj_type in ["str", "list", "dict", "set", "tuple"]:
                    if not self._is_builtin_method_for_type(obj_type, method_name):
                        self.add_error(
                            f"метод '{method_name}' не существует для типа '{obj_type}'",
                            scope_idx,
                            node_idx,
                        )
                elif obj_type in self.builtin_functions:
                    # Это встроенная функция, а не объект
                    continue
                else:
                    # Неизвестный тип - возможно, это ошибка в другом месте
                    pass

    def _method_exists_in_class_hierarchy(
        self, class_name: str, method_name: str
    ) -> bool:
        """Проверяет, существует ли метод в классе или его иерархии наследования"""
        # Сначала ищем среди встроенных методов стандартных типов
        if class_name in ["str", "list", "dict", "set", "tuple"]:
            return self._is_builtin_method_for_type(class_name, method_name)

        visited = set()

        def search_in_class(cls_name):
            if cls_name in visited:
                return False
            visited.add(cls_name)

            # Находим класс
            class_scope = None
            for s in self.all_scopes:
                if (
                    s.get("type") == "class_declaration"
                    and s.get("class_name") == cls_name
                ):
                    class_scope = s
                    break

            if not class_scope:
                # Проверяем, не является ли это встроенным типом
                if cls_name in [
                    "str",
                    "int",
                    "bool",
                    "float",
                    "list",
                    "dict",
                    "set",
                    "tuple",
                ]:
                    return self._is_builtin_method_for_type(cls_name, method_name)
                return False

            # Проверяем методы текущего класса
            for method in class_scope.get("methods", []):
                if method.get("name") == method_name:
                    return True

            # Проверяем статические методы
            for method in class_scope.get("static_methods", []):
                if method.get("name") == method_name:
                    return True

            # Проверяем методы класса (classmethod)
            for method in class_scope.get("class_methods", []):
                if method.get("name") == method_name:
                    return True

            # Рекурсивно проверяем базовые классы
            for base_class in class_scope.get("base_classes", []):
                if search_in_class(base_class):
                    return True

            return False

        return search_in_class(class_name)

    def _extract_method_calls_from_ast(
        self, ast: Dict, method_calls: list, node_idx: int, content: str = ""
    ):
        """Извлекает вызовы методов из AST"""
        if not isinstance(ast, Mapping):
            return

        node_type = ast.get("type")

        if node_type == "method_call":
            obj_name = ast.get("object", "")
            method_name = ast.get("method", "")

            if obj_name and method_name:
                method_calls.append(
                    {
                        "obj": obj_name,
                        "method": method_name,
                        "node_idx": node_idx,
                        "content": content,
                    }
                )

            # Рекурсивно проверяем аргументы
            for arg in ast.get("arguments", []):
                self._extract_method_calls_from_ast(
                    arg, method_calls, node_idx, content
                )

        elif node_type == "function_call":
            # Рекурсивно проверяем аргументы функций
            for arg in ast.get("arguments", []):
                self._extract_method_calls_from_ast(
                    arg, method_calls, node_idx, content
                )

        elif node_type == "binary_operation":
            self._extract_method_calls_from_ast(
                ast.get("left"), method_calls, node_idx, content
            )
            self._extract_method_calls_from_ast(
                ast.get("right"), method_calls, node_idx, content
            )

        elif node_type == "unary_operation":
            self._extract_method_calls_from_ast(
                ast.get("operand"), method_calls, node_idx, content
            )

        elif node_type == "ternary_operator":
            self._extract_method_calls_from_ast(
                ast.get("condition"), method_calls, node_idx, content
            )
            self._extract_method_calls_from_ast(
                ast.get("true_expr"), method_calls, node_idx, content
            )
            self._extract_method_calls_from_ast(
                ast.get("false_expr"), method_calls, node_idx, content
            )

        elif node_type == "list_literal":
            for item in ast.get("items", []):
                self._extract_method_calls_from_ast(
                    item, method_calls, node_idx, content
                )

        elif node_type == "tuple_literal":
            for item in ast.get("items", []):
                self._extract_method_calls_from_ast(
                    item, method_calls, node_idx, content
                )

    def _is_builtin_method_for_type(self, type_name: str, method_name: str) -> bool:
        """Проверяет, является ли метод встроенным для данного типа"""
        builtin_methods = {
            "str": [
                "upper",
                "lower",
                "split",
                "strip",
                "replace",
                "find",
                "startswith",
                "endswith",
                "isdigit",
                "isalpha",
                "format",
                "join",
                "capitalize",
            ],
            "list": [
                "append",
                "extend",
                "insert",
                "remove",
                "pop",
                "clear",
                "index",
                "count",
                "sort",
                "reverse",
                "copy",
            ],
            "dict": [
                "get",
                "keys",
                "values",
                "items",
                "update",
                "pop",
                "clear",
                "copy",
            ],
            "set": [
                "add",
                "remove",
                "discard",
                "pop",
                "clear",
                "union",
                "intersection",
                "difference",
                "copy",
            ],
            "tuple": ["count", "index"],
        }

        if type_name == "Tensor" or type_name.startswith("Tensor["):
            return method_name in {
                "get", "set", "fill", "sum", "copy", "transpose", "matmul",
                "add", "sub", "mul", "div", "add_scalar", "sub_scalar",
                "mul_scalar", "div_scalar", "reshape", "row", "column", "slice",
                "to",
                "shape", "ndim", "size", "device", "release",
            }

        if type_name in builtin_methods:
            return method_name in builtin_methods[type_name]

        return False

    def _add_class_to_registry(self, class_scope: Dict):
        """Добавляет класс в реестр классов"""
        class_name = class_scope.get("class_name")
        if not class_name or class_name in self.classes:
            return

        # Собираем информацию о методах класса
        methods_info = []
        for method in class_scope.get("methods", []):
            methods_info.append(
                {
                    "name": method.get("name"),
                    "is_static": method.get("is_static", False),
                    "is_classmethod": method.get("is_classmethod", False),
                    "parameters": method.get("parameters", []),
                    "return_type": method.get("return_type", ""),
                }
            )

        self.classes[class_name] = {
            "name": class_name,
            "key": "class",
            "type": "class",
            "value": None,
            "id": class_name,
            "is_deleted": False,
            "methods": methods_info,
            "attributes": class_scope.get("attributes", []),
            "static_methods": class_scope.get("static_methods", []),
            "class_methods": class_scope.get("class_methods", []),
            "base_classes": class_scope.get("base_classes", []),
        }

        # Также добавляем в scope_symbols
        level = class_scope.get("level", 0)
        if level not in self.scope_symbols:
            self.scope_symbols[level] = {}
        self.scope_symbols[level][class_name] = self.classes[class_name]
