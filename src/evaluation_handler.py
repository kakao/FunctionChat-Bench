import os
import json
from tqdm import tqdm

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = '/'.join(CUR_PATH.split('/')[:-1])

from src import utils
from src.api_executor import (
    OpenaiModelAzureAPI,
    OpenaiModelAPI
)
from src.formatter import (
    CommonResponseFormatter,
    DialogResponseFormatter,
    SingleCallResponseFormatter,
)
from src.evaluation_registor import (
    CommonEvaluationRegistor,
    DialogEvaluationRegistor,
    SingleCallEvaluationRegistor
)

RESPONSE_FORMATTER_OBJ = {
    'common': CommonResponseFormatter,
    'singlecall': SingleCallResponseFormatter,
    'dialog': DialogResponseFormatter,
}

EVAlUATION_REGISTOR_OBJ = {
    'common': CommonEvaluationRegistor,
    'singlecall': SingleCallEvaluationRegistor,
    'dialog': DialogEvaluationRegistor,
}


class EvaluationHandler:
    """
    A class to handle different types of evaluations for models.
    It manages the setup, execution, and storage of evaluation results based on evaluation metrics and configurations.
    """
    def __init__(self, evaluation_type):
        """
        Initializes the EvaluationHandler with a specific type of evaluation.

        Parameters:
            evaluation_type (str): The type of evaluation to perform, which determines the evaluation logic and outputs.

        Attributes:
            evaluation_type (str): Stores the type of evaluation.
            rubric_prompts (list): Contains the rubric prompts loaded based on evaluation type.
            temperature (float): The temperature setting for model predictions, loaded from configuration.
            executor (object): The API executor instance used to run model predictions.
            eval_reg (object): An instance of the evaluation register object for storing and managing evaluation results.
        """
        self.evaluation_type = evaluation_type
        # load prompt
        self.rubric_prompts = self.get_rubric_prompts()
        cfg = json.loads(open(f'{REPO_PATH}/config/openai.cfg', 'r').read())
        self.temperature = float(cfg.get('temperature'))
        self.executor = self.load_api_executor(cfg)
        self.eval_reg = EVAlUATION_REGISTOR_OBJ[self.evaluation_type]()

    def get_rubric_prompts(self):
        rubric_prompts = {}
        for output_type in ['call', 'completion', 'relevance', 'slot']:
            rubric_file_path = f'{REPO_PATH}/data/rubric_{output_type}.txt'
            if os.path.isfile(rubric_file_path):
                rubric_prompts[output_type] = open(rubric_file_path, "r", encoding="utf-8").read().strip()
        return rubric_prompts

    def load_api_executor(self, cfg):
        executor = None
        # set evaluation-model (default : gpt-4 azure)
        if cfg.get('api_type') == "azure":
            executor = OpenaiModelAzureAPI(cfg.get('instance'), cfg.get('api_key'), cfg.get('api_base'), cfg.get('api_version'))
        elif cfg.get('api_type') == "openai":
            executor = OpenaiModelAPI(cfg.get('api_version'), cfg.get('api_key'), use_eval=True)
        else:
            raise Exception("Unsupported evaluation api type")
        return executor

    def clean_tool_calls(self, tools):
        if not tools:
            return tools
        for tool in tools:
            if 'id' in tool:
                del tool['id']
        return tools

    def get_input_prompt(self, inp, out):
        ground_truth = inp['ground_truth']
        answer_tool_calls = self.clean_tool_calls(ground_truth.get('tool_calls', None))
        if answer_tool_calls:
            ground_truth['tool_calls'] = answer_tool_calls
        out['tool_calls'] = self.clean_tool_calls(out.get('tool_calls', None))
        # create rubric evaluation prompt
        output_type = inp['type_of_output']
        rubric_prompt = self.rubric_prompts.get(output_type, None)
        if rubric_prompt is None:
            raise Exception("Unsupported rubric prompt type")
        tools = json.dumps(inp['tools'], ensure_ascii=False)
        query = json.dumps(inp['messages'], ensure_ascii=False)
        ground_truth = json.dumps(ground_truth, ensure_ascii=False)
        response = json.dumps(out, ensure_ascii=False)
        if output_type == 'call':
            acceptable_arguments = json.dumps(inp['acceptable_arguments'], ensure_ascii=False)
            return rubric_prompt.format(tools=tools,
                                        query=query,
                                        ground_truth=ground_truth,
                                        acceptable_arguments=acceptable_arguments,
                                        response=response)
        elif output_type in ['completion', 'relevance', 'slot']:
            return rubric_prompt.format(tools=tools,
                                        query=query,
                                        ground_truth=ground_truth,
                                        response=response)
        else:
            raise Exception("Unsupported rubric prompt type")

    def get_acceptable_arguments(self, inp):
        acceptable_arguments = inp.get('acceptable_arguments', None)
        if acceptable_arguments:
            try:
                acceptable_arguments = json.loads(acceptable_arguments)
            except Exception:
                acceptable_arguments = json.loads(f'"{acceptable_arguments}"')
        if acceptable_arguments is None:
            return {}
        if acceptable_arguments == "Only ground truth is allowed.":
            return {}
        if acceptable_arguments == "The date should be expressed as 'tomorrow'. A specific date should not be designated.":
            return {}
        if acceptable_arguments == "Since the user did not mention a specific year, it will fail if the date was created including the year in the submission.":
            return {}
        if isinstance(acceptable_arguments, str):
            acceptable_arguments = json.loads(acceptable_arguments)
        return acceptable_arguments

    def compare_arguments(self, g_func_args, p_func_args, acceptable_arguments):
        def compare_value(val1, val2):
            if isinstance(val1, str) and isinstance(val2, str):
                val1 = val1.replace(' ', '').lower()
                val2 = val2.replace(' ', '').lower()
            return val1 == val2

        if g_func_args == p_func_args:
            return True
        j_g_func_args = json.loads(g_func_args)
        try:
            j_p_func_args = json.loads(p_func_args)
        except Exception as e:
            print(f"error : load to json {e}")
            return False
        # argument 할루시네이션
        for key, val in j_p_func_args.items():
            if key not in j_g_func_args:
                return False
        pass_arguments = []
        for key, answer in j_g_func_args.items():
            try:
                predict = j_p_func_args.get(key, None)
            except Exception:
                # predict 가 정상적이지 않음
                return False
            if answer is not None and predict is None:
                return False
            if compare_value(predict, answer) is False:
                if acceptable_arguments:
                    if key in acceptable_arguments:
                        if isinstance(acceptable_arguments[key], list):
                            for acc_answer in acceptable_arguments[key]:
                                if compare_value(predict, acc_answer) is False:
                                    continue
                                else:
                                    pass_arguments.append(key)
                                    break
                        elif isinstance(acceptable_arguments[key], str):
                            acc_answer = acceptable_arguments[key]
                            if compare_value(predict, acc_answer) is False:
                                continue
                            else:
                                pass_arguments.append(key)
            else:
                pass_arguments.append(key)
        if len(pass_arguments) == len(j_g_func_args.keys()):
            return True
        return False

    def match(self, inp, out, debug=False):
        is_pass = "fail"
        fetch_flag = True
        ground_truth = inp.get('ground_truth', {})
        acceptable_arguments = self.get_acceptable_arguments(inp)
        if 'tool_calls' in ground_truth:
            ground_truth = ground_truth.get('tool_calls')[0]['function']
        g_func_name = ground_truth.get('name')
        g_func_args = ground_truth.get('arguments')
        p_func_name = ''
        p_func_args = ''
        predict_tools = out.get('tool_calls', [])
        diff_case_msg = ''
        if predict_tools and len(predict_tools) > 0:
            p_tool = predict_tools[0].get('function', {})
            p_func_name = p_tool.get('name')
            p_func_args = p_tool.get('arguments')
            if g_func_name == p_func_name:
                if self.compare_arguments(g_func_args, p_func_args, acceptable_arguments):
                    is_pass = "pass"
                    fetch_flag = False
                else:
                    diff_case_msg += f'g({g_func_args})|p({p_func_args})\nFunction argument extraction failed.\n'
            else:
                diff_case_msg += f'g({g_func_name})|p({p_func_name})\nFunction selection failed.\n'
        msg = f"exact-eval\n{diff_case_msg}\n\n{is_pass}\n{is_pass}\n"
        input_prompt = ""
        # 임의로 포멧 맞춤
        evaluate_response = {
            "id": "exact-match",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": msg,
                    "role": "assistant"
                },
                "function_call": None,
                "tool_calls": None,
            }],
            "exact": is_pass
        }
        return fetch_flag, evaluate_response, input_prompt

    def fetch(self, inp, out, debug=False):
        input_prompt = self.get_input_prompt(inp, out)
        messages = [{'role': 'user', 'content': input_prompt}]
        evaluate_response = self.executor.predict({'temperature': self.temperature, 'messages': messages})
        if debug is True:
            print(f"\nserial_num : {inp['serial_num']}")
            print(f'evaluate_request : {input_prompt}')
            print(f"evaluate_response : {evaluate_response['choices'][0]['message']['content']}\n")
        return evaluate_response, input_prompt

    def load_cached_evaluation_result(self, eval_file_path, max_size):
        if utils.is_exist_file(eval_file_path):
            eval_output = utils.load_to_jsonl(eval_file_path)
            if len(eval_output) == max_size:
                print(f"[[already evaluate]] .. {len(eval_output)}/{max_size}\npath : {eval_file_path}")
                return eval_output
            else:
                print(f"[[continue .. {len(eval_output)}/{max_size}]]\n")
                return eval_output
        return []

    def evaluate(self, input_set, output_set, eval_file_path, eval_log_file_path, reset, sample, debug=False, only_exact=False):
        """
        Perform the evaluation based on input and output sets, and manage caching and logging of results.

        Parameters:
            input_set (list): A list of input data for the model.
            output_set (list): A list of expected output data corresponding to the input data.
            eval_file_path (str): File path where raw evaluation results are stored.
            eval_log_file_path (str): File path where formatted evaluation logs are stored.
            reset (bool): Whether to reset (overwrite) the existing evaluation results.
            sample (bool): If True, perform a quick evaluation on a small sample.
            debug (bool): If True, print detailed debug information during evaluation.

        Process:
            1. Manages evaluation result caching.
            2. Formats inputs and outputs for processing.
            3. Iteratively processes each input/output pair, logging and storing results.
            4. Displays evaluation metrics upon completion.
        """
        # Manage evaluation result caching
        eval_output = []
        write_option = 'w'
        if reset is False:
            eval_output = self.load_cached_evaluation_result(eval_file_path, len(input_set))
            self.eval_reg.set_eval_output(eval_output)
            if len(eval_output) == len(input_set):
                self.eval_reg.display()
                return
            write_option = 'a'
        # Initialize file writers for raw and formatted logs
        eval_raw_fw = open(eval_file_path, write_option)
        eval_tsv_fw = open(eval_log_file_path, write_option)
        # Process each input/output pair
        start_index = self.eval_reg.get_eval_output_length()
        if debug:
            print("[[evaluate]]")
            print(f" ** start index : {start_index} .. (reset is {reset})")
        # reformated inputs first for proper operation of tqdm
        requests = []
        for inp, out in zip(input_set[start_index:], output_set[start_index:]):
            requests.append((inp, out))
        # load cached evaluation result
        for idx, inp_out_tuple in enumerate(tqdm(requests)):
            # inp keys = ['temperature', 'tool_choice', 'messages', 'tools', 'acceptable_arguments', 'answer']
            # out keys = ['content', 'role', 'function_call', 'tool_calls']
            if sample is True and idx == 1:
                break
            inp, out = inp_out_tuple
            # 'else case' is dialog
            inp['type_of_output'] = 'call' if self.evaluation_type == 'singlecall' else inp['type_of_output']
            # default
            evaluate_response, input_prompt = {}, ''
            fetch_flag = True
            if inp['type_of_output'] == 'call':  # exact match
                fetch_flag, evaluate_response, input_prompt = self.match(inp, out)
            if only_exact:
                fetch_flag = False
            if fetch_flag:
                evaluate_response, input_prompt = self.fetch(inp, out)
            else:
                if len(evaluate_response) == 0:
                    evaluate_response = {
                        "id": "exact-match",
                        "choices": [{
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "content": 'skip evaluation',
                                "role": "assistant"
                            },
                            "function_call": None,
                            "tool_calls": None,
                        }],
                        "exact": 'fail'
                    }
            # formatting
            response_formatter = RESPONSE_FORMATTER_OBJ[self.evaluation_type](
                request_model=inp,
                response_model=out,
                evaluate_prompt=input_prompt,
                evaluate_response=evaluate_response
            )
            if self.eval_reg.get_eval_output_length() == 0:
                title = response_formatter.get_tsv_title()
                eval_tsv_fw.write(f"{title}\n")
            # update eval_output
            output_data = response_formatter.to_dict()
            self.eval_reg.add_eval_output(output_data)
            eval_raw_fw.write(f"{json.dumps(output_data, ensure_ascii=False)}\n")
            eval_tsv_fw.write(f"{response_formatter.to_tsv().strip()}\n")
        # Final display of evaluation metrics
        self.eval_reg.display()
        eval_raw_fw.close()
        eval_tsv_fw.close()
        print(f"[[model evaluation file : {eval_log_file_path}]]")
        return
