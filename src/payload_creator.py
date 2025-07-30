import os
import json
from functools import wraps
from typing import Any, Callable
from tqdm import tqdm
from src import utils
from src.formatter import (
    CommonRequestFormatter,
    DialogRequestFormatter,
    SingleCallRequestFormatter,
)


def validate_params(kwargs):
    expected_types = {
        'request_file_path': str,
        'input_file_path': str,
        'system_prompt_file_path': str,
        'reset': bool,
        'tools_type': str
    }
    tools_type_list = ['all', '4_close', '4_random', '8_close', '8_random']
    for key, expected_type in expected_types.items():
        if key == 'tools_type' and 'tools_type' in kwargs:
            if kwargs[key] is None:
                pass
            elif not isinstance(kwargs[key], expected_type):
                raise ValueError(f"Expected type for {key} is {expected_type}, but got {type(kwargs[key])}.")
            elif kwargs[key] not in tools_type_list:
                raise ValueError(f"tools_type must be one of {tools_type_list}.")
        elif key in kwargs and not isinstance(kwargs[key], expected_type):
            raise ValueError(f"Expected type for {key} is {expected_type}, but got {type(kwargs[key])}.")


def type_check(validate: Callable[[Any, Any], None]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            validate(kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class AbstractPayloadCreator:
    """
    An abstract class designed to create payloads for API requests based on provided parameters and system prompts.
    """
    def __init__(self, temperature, max_size, system_prompt_file_path):
        """
        Initializes the payload creator with temperature settings, maximum payload list size, and a path to a system prompt file.

        Parameters:
            temperature (float): Determines the variability of the model's responses.
            max_size (int): Maximum size or number of payloads to maintain.
            system_prompt_file_path (str): Path to a file containing the prompt text used in payloads.
        """
        self.temperature = temperature
        self.max_size = max_size
        self.system_prompt = None
        if system_prompt_file_path:
            self.system_prompt = self.get_prompt_text(system_prompt_file_path)

    def create_payload(self, **kwargs):
        """
        Abstract method to create a payload for an API request. Must be implemented by subclasses.

        Raises:
            NotImplementedError: Indicates that the method needs to be implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_prompt_text(self, file_path):
        """
        Retrieves and returns the prompt text from a specified file.

        Parameters:
            file_path (str): The path to the file containing the prompt text.

        Returns:
            str: The prompt text, stripped of any leading/trailing whitespace.
        """
        prompt = ''
        if file_path and os.path.isfile(file_path):
            with open(file_path, 'r', encoding="utf-8") as ff:
                prompt = ff.read().strip()
        return prompt

    def load_cached_payload(self, request_file_path):
        """
        Loads cached payload list from a specified file if it exists and meets the required size.

        Parameters:
            request_file_path (str): Path to the file containing cached payloads.

        Returns:
            list: A list of cached payloads if they exist; otherwise, an empty list.
        """
        if utils.is_exist_file(request_file_path):
            api_request_list = utils.load_to_jsonl(request_file_path)
            if len(api_request_list) == self.max_size:
                print(f"[[already existed request jsonl file]] ..{len(api_request_list)}\npath : {request_file_path}")
                print(f"[[already existed request jsonl file]] ..{len(api_request_list)}")
                return api_request_list
            print("[[recreate requests jsonl list]]")
        else:
            print("[[create requests jsonl list]]")
        return []


class CommonPayloadCreator(AbstractPayloadCreator):
    def __init__(self, temperature):
        super().__init__(temperature, 0, None)

    @type_check(validate_params)
    def create_payload(self, **kwargs):
        test_set = utils.load_to_jsonl(kwargs['input_file_path'])
        self.max_size = len(test_set)
        api_request_list = []
        if kwargs['reset'] is False:
            api_request_list = self.load_cached_payload(kwargs['request_file_path'])
            if len(api_request_list) == self.max_size:
                return api_request_list
        else:
            print("[[reset!! create requests jsonl file]]")
        # 2. create requests json list
        for idx, test_input in enumerate(tqdm(test_set)):
            # test_input keys = ['serial_num', 'category', 'input_message', 'input_tools', 'type_of_output', 'ground_truth', 'acceptable_arguments']
            serial_num = test_input['serial_num']
            category = test_input['category']
            tools = test_input['input_tools']
            ground_truth = test_input['ground_truth']
            acceptable_arguments = test_input['acceptable_arguments']
            type_of_output = test_input['type_of_output']
            arguments = {}
            arguments['serial_num'] = serial_num
            arguments['category'] = category
            arguments['tools'] = tools
            arguments['ground_truth'] = ground_truth
            arguments['type_of_output'] = type_of_output
            arguments['acceptable_arguments'] = acceptable_arguments
            arguments['messages'] = test_input['input_messages']
            arguments['temperature'] = self.temperature
            arguments['tool_choice'] = 'auto'
            api_request_list.append(CommonRequestFormatter(**arguments).to_dict())
        # 3. write requests jsonl file
        fi = open(kwargs['request_file_path'], 'w')
        for api_request in api_request_list:
            fi.write(f"{json.dumps(api_request, ensure_ascii=False)}\n")
        fi.close()
        return api_request_list


class DialogPayloadCreator(AbstractPayloadCreator):
    def __init__(self, temperature, system_prompt_file_path):
        super().__init__(temperature, 200, system_prompt_file_path)

    @type_check(validate_params)
    def create_payload(self, **kwargs):
        test_set = utils.load_to_jsonl(kwargs['input_file_path'])
        # update input file max_size
        self.max_size = len(test_set)
        # kwargs keys = ['input_file_path', 'request_file_path', 'reset']
        # 1. check to cached file
        api_request_list = []
        if kwargs['reset'] is False:
            api_request_list = self.load_cached_payload(kwargs['request_file_path'])
            if len(api_request_list) == self.max_size:
                return api_request_list
        else:
            print("[[reset!! create requests jsonl file]]")
        # 2. create requests json list
        for idx, test_input in enumerate(tqdm(test_set)):
            # test_input keys = ['dialog_num', 'tools_count', 'tools', 'turns']
            tools = test_input['tools']
            for turn in test_input['turns']:
                messages = [{'role': 'system', 'content': self.system_prompt}]
                messages.extend(turn['query'])
                arguments = {key: turn[key] for key in ['serial_num', 'ground_truth', 'acceptable_arguments', 'type_of_output']}
                arguments['tools'] = tools
                arguments['messages'] = messages
                arguments['temperature'] = self.temperature
                arguments['tool_choice'] = 'auto'
                api_request_list.append(DialogRequestFormatter(**arguments).to_dict())
        # 3. write requests jsonl file
        fi = open(kwargs['request_file_path'], 'w')
        for api_request in api_request_list:
            fi.write(f"{json.dumps(api_request, ensure_ascii=False)}\n")
        fi.close()
        return api_request_list


class SingleCallPayloadCreator(AbstractPayloadCreator):
    def __init__(self, temperature, system_prompt_file_path):
        super().__init__(temperature, 500, system_prompt_file_path)

    @type_check(validate_params)
    def create_payload(self, **kwargs):
        # kwargs keys = ['input_file_path', 'request_file_path', 'reset', 'tools_type']
        test_set = utils.load_to_jsonl(kwargs['input_file_path'])
        # update input file max_size
        self.max_size = len(test_set)
        # 1. check to cached file
        api_request_list = []
        if kwargs['reset'] is False:
            api_request_list = self.load_cached_payload(kwargs['request_file_path'])
            if len(api_request_list) == self.max_size:
                return api_request_list
        else:
            print("[[reset!! create requests jsonl file]]")
        # 2. create requests json list
        for idx, test_input in enumerate(tqdm(test_set)):
            # test_input keys = ['function_num', 'function_name', 'function_info', 'query',
            #                    'ground_truth', 'acceptable_arguments', 'tools']
            # tools type으로 filtering
            tools_list = []
            tools_type = kwargs['tools_type']
            for t in test_input['tools']:
                if tools_type == 'all' or t['type'] == tools_type:
                    tools_list.append((t['content'], t['type']))
            for q_idx, query in enumerate(test_input['query']):
                messages = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': query['content']}]
                for tools, t_type in tools_list:
                    arguments = {
                        'serial_num': query['serial_num'],
                        'messages': messages,
                        'temperature': self.temperature,
                        'tool_choice': 'auto',
                        'tools': tools,
                        'tools_type': t_type,
                        'acceptable_arguments': test_input['acceptable_arguments'][q_idx]['content'],
                        'ground_truth': test_input['ground_truth'][q_idx]['content'],
                    }
                    api_request_list.append(SingleCallRequestFormatter(**arguments).to_dict())
        # 3. write requests jsonl file
        fi = open(kwargs['request_file_path'], 'w')
        for api_request in api_request_list:
            fi.write(f"{json.dumps(api_request, ensure_ascii=False)}\n")
        fi.close()
        print(f"[[model request file : {kwargs['request_file_path']}]]")
        return api_request_list


class PayloadCreatorFactory:
    """
    A factory class for creating specific payload creators based on the type of evaluation.
    """
    @staticmethod
    def get_payload_creator(evaluation_type, temperature, system_prompt_file_path=None):
        """
        Returns an instance of a payload creator based on the specified evaluation type.

        Parameters:
            evaluation_type (str): The type of evaluation, which determines the type of payload creator.
            temperature (float): The variability setting for the model's responses used in the payload.
            system_prompt_file_path (str, optional): Path to the file containing the system prompt for payloads.

        Returns:
            A payload creator instance appropriate for the given evaluation type.

        Raises:
            ValueError: If the specified evaluation type is not supported.
        """
        if evaluation_type == 'common':
            return CommonPayloadCreator(temperature)
        elif evaluation_type == 'dialog':
            return DialogPayloadCreator(temperature, system_prompt_file_path)
        elif evaluation_type == 'singlecall':
            return SingleCallPayloadCreator(temperature, system_prompt_file_path)
        else:
            raise ValueError("Unsupported evaluation type")
