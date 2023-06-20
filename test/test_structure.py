import base64
import unittest

import app.dims_and_facts


class StructureTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.MODULE = app.dims_and_facts
    
    def test_function_exists_create_orders_by_meal_type_age_cuisine_table(self):
        functions = _get_function_names(self.MODULE)
        self.assertIn(
            base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode(),
            functions,
            msg=f"The function "
                f"`{base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode()}` "
                f"is not found, but it was marked as required."
        )
        
    def test_function_signature_match_create_orders_by_meal_type_age_cuisine_table(self):
        functions = _get_function_names(self.MODULE)
        self.assertIn(
            base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode(),
            functions,
            msg=f"The function "
                f"`{base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode()}` "
                f"is not found, but it was marked as required."
        )
        args = _get_function_arg_names(
            self.MODULE,
            base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode()
        )
        self.assertEqual(
            1,
            len(args),
            msg=f"The function "
                f"`{base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode()}` "
                f"should have exactly 1 argument(s)."
        )
        args = _get_function_arg_names(
            self.MODULE,
            base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode()
        )
        self.assertEqual(
            base64.b64decode(b'ZGI=').decode(),
            args[0],
            msg=f"The argument #0 of function "
                f"`{base64.b64decode(b'Y3JlYXRlX29yZGVyc19ieV9tZWFsX3R5cGVfYWdlX2N1aXNpbmVfdGFibGU=').decode()}` "
                f"should be called "
                f"`db`."
        )
        
    def test_function_exists_load_tables(self):
        functions = _get_function_names(self.MODULE)
        self.assertIn(
            base64.b64decode(b'bG9hZF90YWJsZXM=').decode(),
            functions,
            msg=f"The function "
                f"`{base64.b64decode(b'bG9hZF90YWJsZXM=').decode()}` "
                f"is not found, but it was marked as required."
        )
        
    def test_function_signature_match_load_tables(self):
        functions = _get_function_names(self.MODULE)
        self.assertIn(
            base64.b64decode(b'bG9hZF90YWJsZXM=').decode(),
            functions,
            msg=f"The function "
                f"`{base64.b64decode(b'bG9hZF90YWJsZXM=').decode()}` "
                f"is not found, but it was marked as required."
        )
        args = _get_function_arg_names(
            self.MODULE,
            base64.b64decode(b'bG9hZF90YWJsZXM=').decode()
        )
        self.assertEqual(
            2,
            len(args),
            msg=f"The function "
                f"`{base64.b64decode(b'bG9hZF90YWJsZXM=').decode()}` "
                f"should have exactly 2 argument(s)."
        )
        args = _get_function_arg_names(
            self.MODULE,
            base64.b64decode(b'bG9hZF90YWJsZXM=').decode()
        )
        self.assertEqual(
            base64.b64decode(b'dGFibGVzX2Rpcl9wYXRo').decode(),
            args[0],
            msg=f"The argument #0 of function "
                f"`{base64.b64decode(b'bG9hZF90YWJsZXM=').decode()}` "
                f"should be called "
                f"`tables_dir_path`."
        )
        args = _get_function_arg_names(
            self.MODULE,
            base64.b64decode(b'bG9hZF90YWJsZXM=').decode()
        )
        self.assertEqual(
            base64.b64decode(b'dGFibGVz').decode(),
            args[1],
            msg=f"The argument #1 of function "
                f"`{base64.b64decode(b'bG9hZF90YWJsZXM=').decode()}` "
                f"should be called "
                f"`tables`."
        )
        
    def test_function_exists_reduce_dims(self):
        functions = _get_function_names(self.MODULE)
        self.assertIn(
            base64.b64decode(b'cmVkdWNlX2RpbXM=').decode(),
            functions,
            msg=f"The function "
                f"`{base64.b64decode(b'cmVkdWNlX2RpbXM=').decode()}` "
                f"is not found, but it was marked as required."
        )
        
    def test_function_signature_match_reduce_dims(self):
        functions = _get_function_names(self.MODULE)
        self.assertIn(
            base64.b64decode(b'cmVkdWNlX2RpbXM=').decode(),
            functions,
            msg=f"The function "
                f"`{base64.b64decode(b'cmVkdWNlX2RpbXM=').decode()}` "
                f"is not found, but it was marked as required."
        )
        args = _get_function_arg_names(
            self.MODULE,
            base64.b64decode(b'cmVkdWNlX2RpbXM=').decode()
        )
        self.assertEqual(
            1,
            len(args),
            msg=f"The function "
                f"`{base64.b64decode(b'cmVkdWNlX2RpbXM=').decode()}` "
                f"should have exactly 1 argument(s)."
        )
        args = _get_function_arg_names(
            self.MODULE,
            base64.b64decode(b'cmVkdWNlX2RpbXM=').decode()
        )
        self.assertEqual(
            base64.b64decode(b'ZGI=').decode(),
            args[0],
            msg=f"The argument #0 of function "
                f"`{base64.b64decode(b'cmVkdWNlX2RpbXM=').decode()}` "
                f"should be called "
                f"`db`."
        )
        

# === Internal functions, do not modify ===
import inspect

from types import ModuleType
from typing import List


def _get_function_names(module: ModuleType) -> List[str]:
    names = []
    functions = inspect.getmembers(module, lambda member: inspect.isfunction(member))
    for name, fn in functions:
        if fn.__module__ == module.__name__:
            names.append(name)
    return names


def _get_function_arg_names(module: ModuleType, fn_name: str) -> List[str]:
    arg_names = []
    functions = inspect.getmembers(module, lambda member: inspect.isfunction(member))
    for name, fn in functions:
        if fn.__module__ == module.__name__:
            if fn.__qualname__ == fn_name:
                args_spec = inspect.getfullargspec(fn)
                arg_names = args_spec.args
                if args_spec.varargs is not None:
                    arg_names.extend(args_spec.varargs)
                if args_spec.varkw is not None:
                    arg_names.extend(args_spec.varkw)
                arg_names.extend(args_spec.kwonlyargs)
                break
    return arg_names


def _get_class_names(module: ModuleType) -> List[str]:
    names = []
    classes = inspect.getmembers(module, lambda member: inspect.isclass(member))
    for name, cls in classes:
        if cls.__module__ == module.__name__:
            names.append(name)
    return names


def _get_class_function_names(module: ModuleType, cls_name: str) -> List[str]:
    fn_names = []
    classes = inspect.getmembers(module, lambda member: inspect.isclass(member))
    for cls_name_, cls in classes:
        if cls.__module__ == module.__name__:
            if cls_name_ == cls_name:
                functions = inspect.getmembers(
                    cls,
                    lambda member: inspect.ismethod(member)
                    or inspect.isfunction(member),
                )
                for fn_name, fn in functions:
                    fn_names.append(fn.__qualname__)
                break
    return fn_names


def _get_class_function_arg_names(
    module: ModuleType, cls_name: str, fn_name: str
) -> List[str]:
    arg_names = []
    classes = inspect.getmembers(module, lambda member: inspect.isclass(member))
    for cls_name_, cls in classes:
        if cls.__module__ == module.__name__:
            if cls_name_ == cls_name:
                functions = inspect.getmembers(
                    cls,
                    lambda member: inspect.ismethod(member)
                    or inspect.isfunction(member),
                )
                for fn_name_, fn in functions:
                    if fn.__qualname__ == fn_name:
                        args_spec = inspect.getfullargspec(fn)
                        arg_names = args_spec.args
                        if args_spec.varargs is not None:
                            arg_names.extend(args_spec.varargs)
                        if args_spec.varkw is not None:
                            arg_names.extend(args_spec.varkw)
                        arg_names.extend(args_spec.kwonlyargs)
                        break
                break
    return arg_names
