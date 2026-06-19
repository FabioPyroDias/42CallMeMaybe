from src.parser.parser import parse_arguments
from src.loader.loader import load_json
from src.loader.loader import load_functions, load_prompts, load_vocab
from src.models.models import FunctionDefinition, FunctionCall, TestPrompt
from llm_sdk import Small_LLM_Model
import sys
import torch
import json

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

    generated = []
    max_tokens = 100
    current_token = 0
    error = True
    while (current_token < max_tokens):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_token = logits.index(max(logits))
        generated.append(next_token)
        current_token += 1
    result = function_call + model.decode(generated)
    print(result)

if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parse_arguments()
    try:
        functions = load_functions(load_json(args.functions_definition))
        prompts = load_prompts(load_json(args.input))

        model = Small_LLM_Model()
        vocab = load_vocab(load_json(model.get_path_to_vocab_file()))

        generate_function_call(prompts[0], functions, model, vocab)

        """ input_ids = model.encode(prompts[0].prompt).squeeze().tolist()
        generated = []
        max_tokens = 1
        current_token = 0
        error = True
        while (current_token < max_tokens):
            logits = model.get_logits_from_input_ids(input_ids + generated)
            next_token = logits.index(max(logits))



            generated.append(next_token)

            try:
                output = json.loads(model.decode(generated))
                if "prompt" in output and "name" in output and "parameters" in output:
                    error = True
                    break
            except json.JSONDecodeError:
                pass

            print(f"Token {len(generated)}: {next_token} . Current_token {current_token}")
            current_token += 1
        if error:
            print(f"Error: After {max_tokens} iterations, no output was generated")
        else:
            print(f"Result: {model.decode(generated)}")
        print(model.get_path_to_vocab_file()) """
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
