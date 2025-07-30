import os
import json
import time
import logging
from tqdm import tqdm
from typing import Optional, Union

from src import utils
from src import openai_utils
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
from src.constants import COMMON, SINGLECALL, DIALOG, CALL, COMPLETION, RELEVANCE, SLOT
from src.color import GREEN, RESET

RESPONSE_FORMATTER_OBJ = {
    COMMON: CommonResponseFormatter,
    SINGLECALL: SingleCallResponseFormatter,
    DIALOG: DialogResponseFormatter,
}

EVAlUATION_REGISTOR_OBJ = {
    COMMON: CommonEvaluationRegistor,
    SINGLECALL: SingleCallEvaluationRegistor,
    DIALOG: DialogEvaluationRegistor,
}

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = '/'.join(CUR_PATH.split('/')[:-1])


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
        self.openai_model = cfg['api_version']
        self.openai_apikey = cfg['api_key']
        self.max_tokens = cfg['max_tokens']
        self.eval_reg = EVAlUATION_REGISTOR_OBJ[self.evaluation_type]()
        self.meta_log_file = f"{REPO_PATH}/output/.batch_meta_{self.evaluation_type}.jsonl"
        self.batch_file = f"{REPO_PATH}/output/.batch_{self.evaluation_type}.jsonl"
        self.batch_output_file = f"{REPO_PATH}/output/.batch_{self.evaluation_type}_result.jsonl"

    def get_rubric_prompts(self) -> dict:
        rubric_prompts = {}
        for output_type in [CALL, COMPLETION, RELEVANCE, SLOT]:
            rubric_file_path = os.path.join(REPO_PATH, 'data', f'rubric_{output_type}.txt')
            if os.path.isfile(rubric_file_path):
                try:
                    with open(rubric_file_path, "r", encoding="utf-8") as file:
                        rubric_prompts[output_type] = file.read().strip()
                except IOError as e:
                    logging.warning(f"Error reading {rubric_file_path}: {e}")
        return rubric_prompts

    def load_api_executor(self, cfg: dict) -> Union[OpenaiModelAzureAPI, OpenaiModelAPI]:
        api_type = cfg.get('api_type', None)
        
        if api_type == "azure":
            return OpenaiModelAzureAPI(
                instance=cfg.get('instance'),
                api_key=cfg.get('api_key'),
                api_base=cfg.get('api_base'),
                api_version=cfg.get('api_version')
            )
        elif api_type == "openai":
            return OpenaiModelAPI(
                model=cfg.get('api_version'),
                api_key=cfg.get('api_key'),
                use_eval=True
            )
        else:
            raise ValueError(f"Unsupported evaluation API type: {api_type}")

    def clean_tool_calls(self, tools: list) -> Optional[list]:
        if not tools:
            return tools
        return [{k: v for k, v in tool.items() if k != 'id'} for tool in tools]

    def get_input_prompt(self, inp: dict, out: dict) -> str:
        ground_truth = inp['ground_truth']
        ground_truth['tool_calls'] = self.clean_tool_calls(ground_truth.get('tool_calls', None))
        if out is None:
            out = {'tool_calls': []}
        else:
            out['tool_calls'] = self.clean_tool_calls(out.get('tool_calls', None))
        output_type = inp['type_of_output']
        rubric_prompt = self.rubric_prompts.get(output_type)
        if not rubric_prompt:
            raise ValueError(f"Unsupported rubric prompt type: {output_type}")
        tools = json.dumps(inp['tools'], ensure_ascii=False)
        query = json.dumps(inp['messages'], ensure_ascii=False)
        ground_truth_json = json.dumps(ground_truth, ensure_ascii=False)
        response = json.dumps(out, ensure_ascii=False)

        if output_type == CALL:
            acceptable_arguments = json.dumps(inp['acceptable_arguments'], ensure_ascii=False)
            return rubric_prompt.format(
                tools=tools, query=query,
                ground_truth=ground_truth_json,
                acceptable_arguments=acceptable_arguments,
                response=response
            )
        elif output_type in [COMPLETION, RELEVANCE, SLOT]:
            return rubric_prompt.format(
                tools=tools, query=query,
                ground_truth=ground_truth_json,
                response=response
            )
        else:
            raise ValueError(f"Unsupported rubric prompt type: {output_type}")

    def get_acceptable_arguments(self, inp: dict) -> dict:
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

    def compare_arguments(self, g_func_args: str, p_func_args: str, acceptable_arguments: dict) -> bool:
        def compare_value(val1, val2):
            if isinstance(val1, str) and isinstance(val2, str):
                val1, val2 = val1.replace(' ', '').lower(), val2.replace(' ', '').lower()
            return val1 == val2
    
        try:
            j_g_func_args = json.loads(g_func_args)
            j_p_func_args = json.loads(p_func_args)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            return False

        # argument 할루시네이션
        for key, val in j_p_func_args.items():
            if key not in j_g_func_args:
                return False
        for key, val in j_g_func_args.items():
            p_val = j_p_func_args.get(key)
            if not compare_value(p_val, val):
                acceptable_values = acceptable_arguments.get(key, [])
                if isinstance(acceptable_values, list) and not any(compare_value(p_val, acc) for acc in acceptable_values):
                    return False
                if isinstance(acceptable_values, str) and not compare_value(p_val, acceptable_values):
                    return False
        return True

    def _default_evaluate_response(self, message: str = "skip evaluation", exact: str = "fail") -> dict:
        return {
            "id": "exact-match",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": message, "role": "assistant"},
                "function_call": None,
                "tool_calls": None,
            }],
            "exact": exact
        }

    def exact_match(self, inp: dict, out: dict, debug: bool = False) -> tuple[bool, dict, str]:
        input_prompt = ""
        is_pass = "fail"
        if not inp['type_of_output'] == CALL:
            return False, self._default_evaluate_response(), input_prompt

        if out is None:
            return False, self._default_evaluate_response(message="skip evaluation, because model response is None"), input_prompt

        is_pass_bool = False
        ground_truth = inp.get('ground_truth', {})
        acceptable_arguments = self.get_acceptable_arguments(inp)
        diff_case_msg = ""

        
        if 'tool_calls' in ground_truth:
            ground_truth_func = ground_truth.get('tool_calls', [{}])[0].get('function', {})
        else:
            ground_truth_func = ground_truth
            
        g_func_name = ground_truth_func.get('name')
        g_func_args = ground_truth_func.get('arguments')

        predict_tools = out.get('tool_calls', [])
        if predict_tools:
            predicted_func = predict_tools[0].get('function', {})
            p_func_name = predicted_func.get('name')
            p_func_args = predicted_func.get('arguments')

            if g_func_name == p_func_name:
                if self.compare_arguments(g_func_args, p_func_args, acceptable_arguments):
                    is_pass = "pass"
                    is_pass_bool = True
                else:
                    diff_case_msg += f"Function arguments mismatch: g({g_func_args}) | p({p_func_args})\n"
            else:
                diff_case_msg += f"Function name mismatch: g({g_func_name}) | p({p_func_name})\n"

        msg = f"exact-eval\n{diff_case_msg}\n\n{is_pass}\n{is_pass}\n"

        if debug:
            logging.debug(f"\nInput: {inp}\nOutput: {out}")
            logging.debug(f"Evaluation Message: {msg}")

        return is_pass_bool, self._default_evaluate_response(msg, is_pass), input_prompt

    def _get_batch_info_in_cached_meta(self, meta_log_file: str, client) -> tuple[Optional[str], Optional[str]]:
        from types import SimpleNamespace
        if os.path.isfile(meta_log_file):
            data = {}
            with open(meta_log_file, 'r') as f:
                data = json.load(f)
            batch_meta = SimpleNamespace(**data)
            batch = client.batches.retrieve(batch_meta.id)
            return batch.id, batch.status
        return None, None

    def _execute_batch_request(self, batch_file, debug=False):
        import sys
        import itertools
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_apikey)
        batch_id, batch_status = self._get_batch_info_in_cached_meta(self.meta_log_file, client)
        if batch_status is None:
            batch_input_file = client.files.create(
                file=open(batch_file, "rb"),
                purpose="batch"
            )
            batch_input_file_id = batch_input_file.id
            batch_object = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"FunctionChat-Bench {self.evaluation_type} eval job.."
                }
            )
            with open(self.meta_log_file, 'w') as f:
                f.write(json.dumps(batch_object.to_dict(), indent=2))
            batch_id = batch_object.id
            if batch_id is None:
                raise Exception("openai Batch job Error")
        # Waiting
        batch = client.batches.retrieve(batch_id)
        print(f"\n\nbatch {GREEN}{batch_id}{RESET} status .. {batch.status}")
        if batch.status != 'completed':
            print(f"** If you want to check it yourself, click here. https://platform.openai.com/batches/{batch_id}")
            spinner = itertools.cycle(['～o(▽｀ o)         ', '～o(▽｀ o) =3      ', '～o(▽｀ o) =3 =3   ', '～o(▽｀ o) =3 =3 =3'])
            while batch.status != 'completed':
                for _ in range(120):
                    sys.stdout.write(f"\rWaiting for the batch job to finish. Current batch status is [{batch.status}] {next(spinner)}")
                    sys.stdout.flush()
                    time.sleep(0.5)  # 애니메이션 갱신 속도
                batch = client.batches.retrieve(batch_id)
            print(f"\n\nbatch status .. {batch.status}")
        output_file_id = batch.output_file_id
        content = client.files.content(output_file_id)
        with open(self.batch_output_file, "wb") as file:
            for chunk in content.iter_bytes():
                file.write(chunk)
        batch_result = []
        with open(self.batch_output_file, "r") as file:
            for line in file.readlines():
                batch_result.append(json.loads(line.strip()))
        if len(batch_result) == 0:
            raise Exception(f"batch result is empty. check your batch output : https://platform.openai.com/batches/{batch_id}")
        return batch_result

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

    def _process_exact_match(self, input_set, output_set, start_index):
        from itertools import islice
        requests = islice(zip(input_set, output_set), start_index, None)
        outputs = []
        for idx, (inp, out) in enumerate(tqdm(requests, desc="Processing exact eval")):
            if out is None:
                out = {'tool_calls': []}
            inp['type_of_output'] = 'call' if self.evaluation_type == 'singlecall' else inp['type_of_output']
            is_pass, evaluate_response, input_prompt = self.exact_match(inp, out)
            response_formatter = RESPONSE_FORMATTER_OBJ[self.evaluation_type](
                request_model=inp,
                response_model=out,
                evaluate_prompt=input_prompt,
                evaluate_response=evaluate_response
            )
            outputs.append((is_pass, response_formatter))
        return outputs

    def _finalize_evaluation(self, eval_file_path, eval_log_file_path, outputs, model_name, llm_judge_name, model_path, eval_subtype):
        write_option = 'w'
        # update evaluate register
        for idx, response_formatter in outputs:
            self.eval_reg.add_eval_output(response_formatter.to_dict())
        # show evaluate result
        self.eval_reg.display()
        # save evaluate result
        eval_raw_fw = open(eval_file_path, write_option)
        eval_tsv_fw = open(eval_log_file_path, write_option)
        if write_option == 'w':
            title = response_formatter.get_tsv_title()
            eval_tsv_fw.write(f"{title}\n")
        for idx, response_formatter in outputs:
            eval_raw_fw.write(f"{json.dumps(response_formatter.to_dict(), ensure_ascii=False)}\n")
            eval_tsv_fw.write(f"{response_formatter.to_tsv().strip()}\n")
        eval_raw_fw.close()
        eval_tsv_fw.close()

        if os.path.isfile(self.meta_log_file):
            os.remove(self.meta_log_file)
        if os.path.isfile(self.batch_file):
            os.remove(self.batch_file)
        if os.path.isfile(self.batch_output_file):
            os.remove(self.batch_output_file)
        print(f"[[model evaluation file : {eval_log_file_path}]]")

        self._save_evaluation_result(model_name,llm_judge_name, model_path, eval_subtype)
        

    def _create_batch_file(self, batch_file, outputs):
        input_prompts = []
        with open(batch_file, 'w') as fp:
            for idx, (is_pass, response_formatter) in enumerate(tqdm(outputs, desc="Processing make batch file")):
                inp = response_formatter.request_model
                out = response_formatter.response_model
                if not is_pass:
                    custom_id = f"{self.evaluation_type}_{idx}"
                    input_prompt = self.get_input_prompt(inp, out)
                    messages = [{'role': 'user', 'content': input_prompt}]
                    reformed_json = openai_utils.get_openai_batch_format(custom_id, self.openai_model, messages, self.max_tokens)
                    input_prompts.append(input_prompt)
                    fp.write(json.dumps(reformed_json, ensure_ascii=False)+'\n')
        return input_prompts
       
    def _process_rubric_evaluation(self, outputs, is_batch=False):
        if is_batch:
            input_prompts = self._create_batch_file(self.batch_file, outputs)
            batch_result = self._execute_batch_request(self.batch_file)
            # write_file
            for input_prompt, data in zip(input_prompts, batch_result):
                # custom_id = f"{self.evaluation_type}_{idx}"
                idx = int(data['custom_id'].split('_')[-1])
                response_formatter = outputs[idx][1]
                response_formatter.evaluate_prompt = input_prompt
                response_formatter.set_evaluate_response(data['response']['body'])
                outputs[idx] = (True, response_formatter)
        else:
            for idx, (is_pass, response_formatter) in enumerate(tqdm(outputs, desc="Processing rubric eval")):
                inp = response_formatter.request_model
                out = response_formatter.response_model
                if not is_pass:
                    evaluate_response, input_prompt = self.fetch(inp, out)
                    response_formatter = RESPONSE_FORMATTER_OBJ[self.evaluation_type](
                        request_model=inp,
                        response_model=out,
                        evaluate_prompt=input_prompt,
                        evaluate_response=evaluate_response
                    )
                    outputs[idx] = (True, response_formatter)
        return outputs

    def _save_evaluation_result(self, model_name, llm_judge_name, model_path, eval_subtype):
        eval_score_path = f"{REPO_PATH}/output/{model_name}/FunctionChat-{model_name}.eval_score.json"
        if not os.path.isdir(f"{REPO_PATH}/output/{model_name}"):
            os.makedirs(f"{REPO_PATH}/output/{model_name}")
        singlecall_score = {}
        dialog_score = {}
        calldecision_score = {}
        common_score = {}
        if os.path.isfile(eval_score_path):
            with open(eval_score_path, 'r', encoding='utf-8') as f:
                try:
                    total_score = json.load(f)
                    singlecall_score = total_score.get('singlecall_score', {})
                    dialog_score = total_score.get('dialog_score', {})
                    calldecision_score = total_score.get('calldecision_score', {})
                    common_score = total_score.get('common_score', {})
                except Exception as e:
                    print(f"Error loading evaluation score: {e}")
                    singlecall_score = {}
                    dialog_score = {}
                    calldecision_score = {}
                    common_score = {}
        if self.evaluation_type == SINGLECALL:
            singlecall_score = self.eval_reg.get_score()
        elif self.evaluation_type == DIALOG:
            dialog_score = self.eval_reg.get_score()
        elif self.evaluation_type == COMMON:
            if eval_subtype == 'CallDecision':
                calldecision_score = self.eval_reg.get_score()
            else:
                common_score = self.eval_reg.get_score()
        else:
            raise ValueError(f"Unsupported evaluation type: {self.evaluation_type}")
        
        fcb_version, fcb_environments = utils.get_git_info()

        with open(eval_score_path, 'w', encoding='utf-8') as f:
            total_score = {
                'fcb_version': fcb_version,
                'fcb_environments': fcb_environments,
                'llm_judge_model': llm_judge_name,
                'target_model_path': str(model_path),
                'singlecall_score': singlecall_score,
                'dialog_score': dialog_score,
                'calldecision_score': calldecision_score
            }
            if len(common_score) > 0:
                total_score[f'{eval_subtype.lower()}_score'] = common_score
            f.write(json.dumps(total_score, ensure_ascii=False, indent=4))
        print(f"[[evaluation scores saved to: {eval_score_path}]]")

    def _set_batch_file_names(self, model_name=None):
        """
        배치 관련 파일 이름을 설정합니다.

        Parameters:
            model_name (str, optional): include model name in file name.
        """
        if model_name:
            self.meta_log_file = f"{REPO_PATH}/output/.batch_meta_{self.evaluation_type}_{model_name}.jsonl"
            self.batch_file = f"{REPO_PATH}/output/.batch_{self.evaluation_type}_{model_name}.jsonl"
            self.batch_output_file = f"{REPO_PATH}/output/.batch_{self.evaluation_type}_{model_name}_result.jsonl"
        else:
            self.meta_log_file = f"{REPO_PATH}/output/.batch_meta_{self.evaluation_type}.jsonl"
            self.batch_file = f"{REPO_PATH}/output/.batch_{self.evaluation_type}.jsonl"
            self.batch_output_file = f"{REPO_PATH}/output/.batch_{self.evaluation_type}_result.jsonl"

    def evaluate(self, input_set, output_set, eval_file_path, eval_log_file_path, reset, sample,
                 debug=False, only_exact=False, model_name=None, llm_judge_name=None, model_path=None, is_batch=False,
                 eval_subtype=None):
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
            model_name (str): Name of the model being evaluated.
            llm_judge_name (str): Name of the LLM judge used for evaluation.
        """
        if not eval_subtype:
            eval_subtype = self.evaluation_type
        # check cached file
        self._set_batch_file_names(model_name)
        
        self.eval_reg.set_eval_output(
            self.load_cached_evaluation_result(eval_file_path, len(input_set)) if not reset else []
        )
        eval_output_length = self.eval_reg.get_eval_output_length()
        if eval_output_length == len(input_set):
           self.eval_reg.display()
           self._save_evaluation_result(model_name, llm_judge_name, model_path, eval_subtype)
           return
        # start evaluation
        start_time = time.time()
        if debug:
            print("[[evaluate]]")
            print(f" ** start index : {eval_output_length} .. (reset is {reset})")
        if sample:
            # TODO : sample 1개만 실행하고 파일에 저장하게 작업 추가
            return
        outputs = self._process_exact_match(input_set, output_set, eval_output_length)
        if not only_exact:
            outputs = self._process_rubric_evaluation(outputs, is_batch)
        self._finalize_evaluation(eval_file_path, eval_log_file_path, outputs, model_name, llm_judge_name, model_path, eval_subtype)
        elapsed_time = time.time() - start_time
        print(f"Total time execution: {elapsed_time:.2f} seconds")
        return
