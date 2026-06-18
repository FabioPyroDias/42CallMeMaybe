from src.parser.parser import parse_arguments
from src.loader.loader import load_json, load_functions, load_prompts
import sys

if __name__ == "__main__":
    args = parse_arguments()
    try:
        functions = load_functions(load_json(args.functions_definition))
        prompts = load_prompts(load_json(args.input))
        print(functions)
        print(prompts)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)
