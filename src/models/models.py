from pydantic import BaseModel
from typing import Any


class Parameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class TestPrompt(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
