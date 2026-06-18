import argparse
import json
from typing import Any, Union
from src.models.models import FunctionDefinition, TestPrompt


def parse_arguments() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--functions_definition",
                            default="data/input/functions_definition.json")
    arg_parser.add_argument("--input",
                            default="data/input/function_calling_tests.json")
    arg_parser.add_argument("--output",
                            default="data/output/function_calls.json")
    return arg_parser.parse_args()
