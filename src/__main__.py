from src.parser.parser import parse_arguments
from src.loader.loader import load_json
from src.loader.loader import load_functions, load_prompts, load_vocab
from src.models.models import FunctionDefinition, FunctionCall, TestPrompt
from llm_sdk import Small_LLM_Model
import sys
from numpy import inf
import json



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
                        for token_end in field_end:
                            if token_id == token_end:
                                exists = True
                elif token_id == function[len(generated)]:
                    exists = True

            if not exists:
                logits[token_id] = -inf

        next_token = logits.index(max(logits))
        generated.append(next_token)

        for finish in field_end:
            if next_token == finish:
                return generated

        for index in range(len(possible_tokens) - 1, -1, -1):
            if len(generated) >= len(possible_tokens[index]):
                continue
            elif possible_tokens[index][len(generated) - 1] != next_token:
                possible_tokens.remove(possible_tokens[index])

        current_token += 1

    return []



def constrained_decoding(model: Small_LLM_Model, input_ids: list[int],
                         function_call: str,
                         functions: list[FunctionDefinition]) -> str:
    functions_ids = []
    for function in functions:
        functions_ids.append(model.encode(function.name).squeeze().tolist())

    field_end = model.encode("\"").squeeze().tolist()
    if isinstance(field_end, int):
        field_end = [field_end]

    generated = generate_field(model, input_ids, functions_ids, field_end)

    if len(generated) == 0:
        raise ValueError("Could not generate a valid function name "\
                         "within the token limit")

    return function_call + model.decode(generated)

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

    function_call = f"{{\"prompt\": \"{prompt.prompt}\",\n\"name\": \""
    input_ids = model.encode(encoding_message + function_call).squeeze().tolist()

    constrained_decoding(model, input_ids, function_call, functions)


if __name__ == "__main__":
    args = parse_arguments()
    try:
        functions = load_functions(load_json(args.functions_definition))
        prompts = load_prompts(load_json(args.input))

        model = Small_LLM_Model()
        vocab = load_vocab(load_json(model.get_path_to_vocab_file()))

        generate_function_call(prompts[0], functions, model, vocab)

    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
