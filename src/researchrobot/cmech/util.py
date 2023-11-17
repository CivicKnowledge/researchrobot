import inspect
from docstring_parser import parse
from pydantic import BaseModel
from typing import Any, Dict, Type, List, get_type_hints

def map_types(v):
    return {
        'str': 'string',
        'Optional': 'object',
    }.get(v, v)

def get_pydantic_model_spec(model: Type[BaseModel]) -> Dict[str, Any]:
    """Generate specification for Pydantic model fields."""
    spec = {}
    for field_name, field in model.model_fields.items():
        spec[field_name] = {
            "type": map_types(field.annotation.__name__),
            "description": field.description
        }
    return spec

def generate_tools_specification(cls: Type) -> List[Dict[str, Any]]:
    """
    Generates a tools specification from a class's methods, annotations, and docstrings
    using the docstring_parser module. It excludes the 'self' parameter from methods.
    """
    functions_spec = []

    for name, method in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith('__'):
            continue

        sig = inspect.signature(method)
        docstring = method.__doc__
        parsed_docstring = parse(docstring) if docstring else None

        function_spec = {
            "name": name,
            "description": parsed_docstring.short_description if parsed_docstring else '',
            "parameters": {
                "type": "object",
                "properties": {}
            },
            "result": {
                "type": "object",
                "description": parsed_docstring.returns.description if parsed_docstring and parsed_docstring.returns else '',
                "properties": {}
            }
        }

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue  # Skip the 'self' parameter

            param_type = param.annotation
            if param_type == param.empty:
                param_type = Any
            else:
                param_type = get_type_hints(method)[param_name]

            param_doc = next((p for p in parsed_docstring.params if p.arg_name == param_name), None) if parsed_docstring else None
            param_description = param_doc.description if param_doc else ''

            if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
                function_spec["parameters"]["properties"][param_name] = {
                    "type": map_types(param_type.__name__),
                    "description": param_description,
                    "properties": get_pydantic_model_spec(param_type)
                }
            else:
                function_spec["parameters"]["properties"][param_name] = {
                    "type": map_types(param_type.__name__),
                    "description": param_description
                }

        return_type = sig.return_annotation
        if return_type == sig.empty:
            return_type = Any
        else:
            return_type = get_type_hints(method).get('return', Any)

        if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
            function_spec["result"] = {
                "type": return_type.__name__,
                "properties": get_pydantic_model_spec(return_type)
            }
        else:
            function_spec["result"]["type"] = return_type.__name__ if not isinstance(return_type, str) else return_type

        # Patch up bad formatting
        function_spec = {
            'type': 'function',
            'function': function_spec
        }

        functions_spec.append(function_spec)

    return functions_spec

# Example usage
# class MyExampleClass:
#     def my_method(self, param1: int, param2: MyPydanticModel) -> MyPydanticModel:
#         """
#         Example method description.
#         Args:
#           param1 (int): Description of param1.
#           param2 (MyPydanticModel): Description of param2.
#         Returns:
#           MyPydanticModel: Description of return value.
#         """
#         pass

# tools_spec = generate_tools_specification(MyExampleClass)
# print(tools_spec)
