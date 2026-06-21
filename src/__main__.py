from src.parser.parser import parse_arguments
from src.loader.loader import load_json
from src.loader.loader import load_functions, load_prompts, load_vocab
from src.models.models import FunctionDefinition, FunctionCall, TestPrompt
from llm_sdk import Small_LLM_Model
import sys
from numpy import inf
import json
from enum import Enum
from pydantic import ValidationError


def encode_to_list(model, text: str) -> list[int]:
    result = model.encode(text).squeeze().tolist()
    if isinstance(result, int):
        return [result]
    return result

def decode_tokens(vocab: dict[int, str], tokens: list[int]) -> str:
    decoded = ""
    for token in tokens:
        decoded += vocab[token]
    decoded = decoded.replace("Ġ", " ")
    decoded = decoded.replace("Ċ", "\n")
    return decoded


def generate_field(model, input_ids, samples, field_end) -> list[int]:
    generated = []

    max_tokens = 100
    current_token = 0
    possible_tokens = samples.copy()
    while (current_token < max_tokens):
        logits = model.get_logits_from_input_ids(input_ids + generated)

        for token_id in range(len(logits)):
            exists = False
            for function in possible_tokens:
                if len(generated) >= len(function):
                    if len(function) == len(generated):
                        if token_id == field_end:
                            exists = True
                elif token_id == function[len(generated)]:
                    exists = True

            if not exists:
                logits[token_id] = -inf

        next_token = logits.index(max(logits))
        generated.append(next_token)

        if next_token == field_end:
            return generated

        for index in range(len(possible_tokens) - 1, -1, -1):
            if len(generated) >= len(possible_tokens[index]):
                continue
            elif possible_tokens[index][len(generated) - 1] != next_token:
                possible_tokens.remove(possible_tokens[index])

        current_token += 1

    return []

def get_parameter_token_ids(vocab: dict[int, str], decoded: list[str]) -> list[int]:
    token_ids = []
    for id, value in vocab.items():
        if value in decoded:
            token_ids.append(id)
    return token_ids


class ParameterString(Enum):
    BEGIN = 0
    GENERATING = 1
    END = 2

def get_current_string_invalid(state: ParameterString) -> list[str]:
    if state == ParameterString.BEGIN:
        return ["\""]
    elif state == ParameterString.GENERATING:
        return []
    return []

def get_next_string_state(state: ParameterString, next_token_char: str) -> ParameterString:
    if next_token_char == "\"":
        return ParameterString.END

    if state == ParameterString.BEGIN or state == ParameterString.GENERATING:
        return ParameterString.GENERATING

    raise ValueError(f"Unexpected state '{state.name}' "
                     f"in string parameter generation")

def generate_string_value(model: Small_LLM_Model, vocab: dict[int, str], prompt: list[int]) -> list[int]:
    current_state = ParameterString.BEGIN
    current_token = 0
    max_tokens = 100
    generated = []

    while (current_token < max_tokens):
        impossible_tokens = get_current_string_invalid(current_state)
        invalid_ids = get_parameter_token_ids(vocab, impossible_tokens)
        logits = model.get_logits_from_input_ids(prompt + generated)
        for token_id in range(len(logits)):
            if token_id in invalid_ids:
                logits[token_id] = -inf
        
        next_token = logits.index(max(logits))
        generated.append(next_token)

        decoded_parameter = decode_tokens(vocab, generated)
        terminated_index = decoded_parameter.find("\"")
        if terminated_index != -1:
            decoded_parameter_restricted = decoded_parameter[0: terminated_index + 1]
            return encode_to_list(model, decoded_parameter_restricted)

        current_state = get_next_string_state(current_state, vocab[next_token])

        if current_state == ParameterString.END:
            return generated

        current_token += 1

    return []

class ParameterNumber(Enum):
    BEGIN = 0
    FIRST_DIGIT = 1
    AFTER_FIRST_DIGIT = 2
    AFTER_POINT = 3
    AFTER_DECIMAL = 4
    END = 5

def get_current_number_parameter(state: ParameterNumber, number_type: str, is_last: bool):
    if state == ParameterNumber.BEGIN:
        return ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif state == ParameterNumber.FIRST_DIGIT:
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif state == ParameterNumber.AFTER_FIRST_DIGIT:
        if number_type == "integer":
            if is_last:
                return ["}", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            else:
                return [",", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        else:
            return [".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif state == ParameterNumber.AFTER_POINT:
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif state == ParameterNumber.AFTER_DECIMAL:
        if is_last:
            return ["}", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        else:
            return [",", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return []

def get_next_number_state(state, next_token_char) -> ParameterNumber:
    if next_token_char == "}" or next_token_char == ",":
        return ParameterNumber.END

    if next_token_char == "-":
        return ParameterNumber.FIRST_DIGIT

    if state == ParameterNumber.BEGIN:
        return ParameterNumber.AFTER_FIRST_DIGIT

    if state == ParameterNumber.FIRST_DIGIT:
        return ParameterNumber.AFTER_FIRST_DIGIT

    if next_token_char == ".":
        return ParameterNumber.AFTER_POINT

    if state == ParameterNumber.AFTER_FIRST_DIGIT:
        return ParameterNumber.AFTER_FIRST_DIGIT

    if state == ParameterNumber.AFTER_POINT:
        return ParameterNumber.AFTER_DECIMAL

    if state == ParameterNumber.AFTER_DECIMAL:
        return ParameterNumber.AFTER_DECIMAL

    raise ValueError(f"Unexpected character '{next_token_char}' "
                     f"in state {state}")

def generate_number_value(model: Small_LLM_Model, vocab: dict[int, str], prompt: list[int], number_type: str, is_last: bool) -> list[int]:
    current_state = ParameterNumber.BEGIN
    current_token = 0
    max_tokens = 100
    generated = []

    while current_token < max_tokens:
        possible_tokens = get_current_number_parameter(current_state, number_type, is_last)
        input_ids = get_parameter_token_ids(vocab, possible_tokens)
        logits = model.get_logits_from_input_ids(prompt + generated)
        for token_id in range(len(logits)):
            if token_id not in input_ids:
                logits[token_id] = -inf

        next_token = logits.index(max(logits))
        generated.append(next_token)
        current_state = get_next_number_state(current_state, vocab[next_token])

        if current_state == ParameterNumber.END:
            if vocab[generated[-1]] == "}":
                generated.pop()
            return generated

        current_token += 1

    return []



def constrained_decoding(model: Small_LLM_Model, input_ids: list[int],
                         function_call_tokens: list[int],
                         functions: list[FunctionDefinition],
                         vocab: dict[int, str]) -> str:
    functions_ids = []
    for function in functions:
        functions_ids.append(encode_to_list(model, function.name))

    field_end = None
    for key, value in vocab.items():
        if value == "\"":
            field_end = key
            break

    if not field_end:
        raise ValueError("No terminating field character found")

    generated = generate_field(model, input_ids, functions_ids, field_end)

    if len(generated) == 0:
        raise ValueError("Could not generate a valid function name "\
                         "within the token limit")

    function_call_tokens += generated
    function_call_tokens += encode_to_list(model,",\n\"parameters\": {")

    function_name = decode_tokens(vocab, generated)
    function_name = function_name[0 : -1]
    current_function = None
    for function in functions:
        if function.name == function_name:
            current_function = function
            break
    if not current_function:
        raise ValueError("Generated function name does not match " \
                         "known functions")

    parameter_index = 0
    for name, value in current_function.parameters.items():
        function_call_tokens += encode_to_list(model, f"\"{name}\": ")

        parameter_index += 1
        if value.type == "number" or value.type == "integer":
            function_call_tokens += generate_number_value(model, vocab, function_call_tokens, value.type, parameter_index == len(current_function.parameters.keys()))
        elif value.type == "string":
            function_call_tokens += encode_to_list(model, "\"")
            function_call_tokens += generate_string_value(model, vocab, function_call_tokens)
        else:
            raise ValueError(f"Undefined type '{value.type}' in Parameters")

        if parameter_index == len(current_function.parameters.keys()):
            function_call_tokens += encode_to_list(model, "}\n")

    function_call_tokens += encode_to_list(model, "}")

    return decode_tokens(vocab, function_call_tokens)

def generate_function_call(prompt: TestPrompt,
                           functions: list[FunctionDefinition],
                           model: Small_LLM_Model,
                           vocab: dict[int, str]) -> FunctionCall:
    encoding_message = f"Prompt: {prompt.prompt}\nFunctions: [ "
    index = 0
    while index < len(functions):
        encoding_message += f"[ Name: {functions[index].name} , "
        encoding_message += f"Description: {functions[index].description} , "
        encoding_message += f"Parameters: [ "
        params_list = list(functions[index].parameters.items())
        param_index = 0
        while param_index < len(params_list):
            key, value = params_list[param_index]
            encoding_message += f"key: {key} , value: {value.type}"
            param_index += 1
            if (param_index < len(params_list)):
                encoding_message += " ; "
        encoding_message += " ] ]"
        index += 1
        if index < len(functions):
            encoding_message += " , \n "
    encoding_message += " ]"

    function_call = f"{{\n\"prompt\": \"{prompt.prompt}\",\n\"name\": \""
    function_call_tokens = encode_to_list(model, function_call)
    input_ids = encode_to_list(model, encoding_message) + function_call_tokens

    function_call = constrained_decoding(model, input_ids, function_call_tokens, functions, vocab)

    try:
        body = json.loads(function_call)
        return FunctionCall.model_validate(body)
    except (json.JSONDecodeError, ValidationError):
        raise ValueError(f"Function call {function_call} incorrectly structured")

if __name__ == "__main__":
    args = parse_arguments()
    try:
        functions = load_functions(load_json(args.functions_definition))
        prompts = load_prompts(load_json(args.input))

        model = Small_LLM_Model()
        vocab = load_vocab(load_json(model.get_path_to_vocab_file()))

        generate_function_call(prompts[2], functions, model, vocab)

    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
