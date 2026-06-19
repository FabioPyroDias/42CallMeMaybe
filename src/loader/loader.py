import json
from typing import Any
from src.models.models import FunctionDefinition, TestPrompt
from pydantic import ValidationError


def load_json(path: str) -> list[dict[Any, Any]]:
    try:
        with open(path) as fd:
            return json.load(fd)
    except PermissionError:
        raise ValueError("No permissions to open the file")
    except FileNotFoundError:
        raise ValueError("File does not exist")
    except IsADirectoryError:
        raise ValueError("Expected file, received directory")
    except json.JSONDecodeError:
        raise ValueError("JSON file not properly formatted")


def load_functions(f: list[dict[Any, Any]]) -> list[FunctionDefinition]:
    try:
        return [FunctionDefinition.model_validate(function)
                for function in f]
    except ValidationError as error:
        raise ValueError(f"Invalid function definition: {error}")


def load_prompts(prompts: list[dict[Any, Any]]) -> list[TestPrompt]:
    try:
        return [TestPrompt.model_validate(prompt) for prompt in prompts]
    except ValidationError as error:
        raise ValueError(f"Invalid prompt definition: {error}")


def load_vocab(vocab: dict[Any, Any]) -> dict[int, str]:
    new_vocab = {}
    for key, value in vocab.items():
        new_vocab[value] = key
    return new_vocab
