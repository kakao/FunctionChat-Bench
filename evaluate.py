#!/usr/bin/env python3

import os
import sys
import click
import inspect
import json

from pathlib import Path

from src import utils
from src import local_inference
from src.default_click_type import (
    DefaultBaseUrlPromptOptions,
    DefaultModelPathPromptOptions,
    DefaultResetPromptOptions,
    DefaultSamplePromptOptions,
    DefaultDebugPromptOptions,
    DefaultGPidPromptOptions,
    DefaultGLocPromptOptions,
    DefaultApiKeyPromptOptions,
    DefaultToolParserPromptOptions,
    DefaultServedModelNamePromptOptions,
    DefaultServingWaitTimeoutPromptOptions,
    DefaultBatchPromptOptions,
    DefaultOnlyExactPromptOptions
)

from src.payload_creator import PayloadCreatorFactory
from src.response_handler import ResponseHandler
from src.evaluation_handler import EvaluationHandler
from src.constants import DEFAULT_TEMPERATURE, LOCALHOST_BASE_URL, EXIT_SUCCESS, EXIT_FAILURE


REPO_PATH = os.path.dirname(os.path.abspath(__file__))


# program options
@click.group()
@click.option("-q", help="disable all prompts", flag_value=True, default=True)
@click.pass_context
def cli(ctx, q):
    ctx.ensure_object(dict)
    ctx.obj['q'] = q


def default_eval_options(f):
    f = click.option('--model', prompt='model name', help='gpt-3.5-turbo, gpt-4 ..etc')(f)
    f = click.option('--input_path', prompt='input file path', help='golden set file name (*.jsonl)')(f)
    # test option
    f = click.option('--reset', prompt='recreate request file', help='reset request file', cls=DefaultResetPromptOptions)(f)
    f = click.option('--sample', prompt='Run only 1 case.', help='run sample', cls=DefaultSamplePromptOptions)(f)
    f = click.option('--debug', prompt='debug flag', help='debugging', cls=DefaultDebugPromptOptions)(f)
    f = click.option('--only_exact', prompt='evaluate exact match', help='only exact match(True, False)', cls=DefaultOnlyExactPromptOptions)(f)
    f = click.option('--is_batch', prompt='batch processing', help='batch processing(True, False)', cls=DefaultBatchPromptOptions)(f)
    # openai type
    f = click.option('--api_key', prompt='model api key', help='api key', cls=DefaultApiKeyPromptOptions)(f)
    f = click.option('--temperature', prompt='temperature', help='generate temperature', default=DEFAULT_TEMPERATURE)(f)
    # openai - hosting server type
    f = click.option('--base_url', prompt='model api url', help='base url', cls=DefaultBaseUrlPromptOptions)(f)
    f = click.option('--served_model_name', prompt='served model name', help='gpt-3.5-turbo, gpt-4 ..etc', cls=DefaultServedModelNamePromptOptions)(f)
    # gemini
    f = click.option('--gcloud_project_id', prompt='gemini project id', help='google pid', cls=DefaultGPidPromptOptions)(f)
    f = click.option('--gcloud_location', prompt='gemini location', help='google cloud location', cls=DefaultGLocPromptOptions)(f)
    # for local inference
    f = click.option('--model_path', prompt='inhouse model path', help='model path in header', cls=DefaultModelPathPromptOptions)(f)
    f = click.option('--serving_wait_timeout', prompt='serving_wait_timeout', help='--serving-wait-timeout', cls=DefaultServingWaitTimeoutPromptOptions)(f)
    f = click.option('--tool_parser', prompt='tool_parser', help='--tool-call-parser (like functionary_llama_v3)', cls=DefaultToolParserPromptOptions)(f)
    return f


def dialog_eval_options(f):
    f = click.option('--system_prompt_path', prompt='system_prompt_path', help='system prompt file path')(f)
    return f


def singlecall_eval_options(f):
    f = click.option('--system_prompt_path', prompt='system_prompt_path', help='system prompt file path')(f)
    f = click.option('--tools_type', prompt='tools type', help='tools_type = {exact, 4_random, 4_close, 8_random, 8_close}')(f)
    return f


def get_file_paths(test_prefix, model_name, tools_type=None):
    output_path = f'{REPO_PATH}/output/'
    utils.create_directory(output_path)
    
    # 모델별 디렉토리 생성
    model_output_path = f'{output_path}/{model_name}'
    utils.create_directory(model_output_path)
    print(f"output_path: {model_output_path}")
    
    base_path = f'{model_output_path}/{test_prefix}'
    if tools_type:
        return {
            "request": f"{base_path}.input.jsonl",
            "predict": f"{base_path}.{model_name}.{tools_type}.output.jsonl",
            "eval": f"{base_path}.{model_name}.{tools_type}.eval.jsonl",
            "eval_log": f"{base_path}.{model_name}.{tools_type}.eval_report.tsv",
        }
    else:
        return {
            "request": f"{base_path}.input.jsonl",
            "predict": f"{base_path}.{model_name}.output.jsonl",
            "eval": f"{base_path}.{model_name}.eval.jsonl",
            "eval_log": f"{base_path}.{model_name}.eval_report.tsv",
        }


def get_eval_subtype(eval_type, input_path):
    import os
    from src.constants import SINGLECALL, DIALOG, COMMON
    if eval_type == SINGLECALL:
        return eval_type
    elif eval_type == DIALOG:
        return eval_type
    elif eval_type == COMMON:
        filename = os.path.basename(input_path)
        # ex : FunctionChat-CallDecision.jsonl -> CallDecision
        subtype = filename.split('FunctionChat-')[-1].split('.jsonl')[0]
        return subtype
    else:
        return eval_type

def run_evaluate(
        eval_type,
        test_prefix,
        model,
        input_path,
        temperature, api_key, 
        # inhouse
        base_url, served_model_name, 
        # inhouse-local
        model_path, tool_parser, serving_wait_timeout,
        reset, sample, debug, only_exact,
        gcloud_project_id, gcloud_location,
        system_prompt_path=None, # common일때 불필요
        tools_type=None, # singlecall 일때만 필요
        is_batch=True, # batch processing 옵션
    ):
    eval_subtype = get_eval_subtype(eval_type, input_path)
    model_name = None
    if model == 'inhouse-local':
        model_name = Path(model_path).name
    elif model == 'inhouse':
        model_name = served_model_name
    else: # gpt, mistral, etc.
        model_name = model

    file_paths = get_file_paths(test_prefix, model_name)
    print(f"[[{model_name} {test_prefix} evaluate start]]")
    process_meta = None
    try:
        # 파일이 없거나, predict 라인수가 requests 보다 작을때만 모델을 띄우게 
        if not utils.compare_file_line_counts(file_paths['request'], file_paths['predict']):
            if model == 'inhouse-local':
                if Path(model_path).exists():
                    process_meta = local_inference.initialize_vllm(
                                        model_path,
                                        model_name,
                                        tool_parser,
                                        serving_wait_timeout
                                   )
                    api_key = model
                    base_url = LOCALHOST_BASE_URL
                    model_path = model_name # vllm_fc 에서 request.model 에 model_name 만 들어감
                else:
                    raise Exception("Invalid model_path")

        api_request_list = PayloadCreatorFactory.get_payload_creator(
            eval_type, temperature,
           system_prompt_path # option arguments
        ).create_payload(
            input_file_path=input_path,
            request_file_path=file_paths['request'],
            reset=reset,
            tools_type=tools_type # option arguments
        )
        api_response_list = ResponseHandler(
            model, api_key, base_url, model_name,
            gcloud_project_id, gcloud_location
        ).fetch_and_save(
            api_request_list, file_paths['predict'], reset, sample, debug
        )
        if process_meta is not None:
            local_inference.kill_vllm(process_meta)
        
        # Get LLM judge name from config
        cfg = json.loads(open(f'{REPO_PATH}/config/openai.cfg', 'r').read())
        llm_judge_name = cfg.get('api_version', 'unknown')
        
        EvaluationHandler(eval_type).evaluate(
            api_request_list, api_response_list,
            file_paths['eval'], file_paths['eval_log'],
            reset, sample, debug, only_exact,
            model_name=model_name,
            llm_judge_name=llm_judge_name,
            model_path=model_path if model == 'inhouse-local' else model_name,
            is_batch=is_batch,
            eval_subtype=eval_subtype
        )
    except KeyboardInterrupt:
        print("Ctrl+C detected. Terminating the process.")
        if process_meta is not None:
            local_inference.kill_vllm(process_meta)
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if process_meta is not None:
            local_inference.kill_vllm(process_meta)
        sys.exit(EXIT_FAILURE)
        


# program command
@cli.command()
@default_eval_options
@dialog_eval_options
def dialog(model,
           input_path, system_prompt_path,
           temperature, api_key,
           # inhouse
           base_url, served_model_name, 
           # inhouse-local
           model_path, tool_parser, serving_wait_timeout,
           reset, sample, debug, only_exact,
           gcloud_project_id, gcloud_location,
           is_batch):
    eval_type = inspect.stack()[0][3]
    run_evaluate(
      eval_type, f'FunctionChat-{eval_type.capitalize()}',
      model,
      input_path,
      temperature, api_key, 
      base_url, served_model_name, 
      model_path, tool_parser, serving_wait_timeout,
      reset, sample, debug, only_exact,
      gcloud_project_id, gcloud_location,
      system_prompt_path=system_prompt_path,
      is_batch=is_batch
    )


@cli.command()
@default_eval_options
@singlecall_eval_options
def singlecall(model, input_path, system_prompt_path,
               temperature, api_key, 
               # inhouse
               base_url, served_model_name, 
               # inhouse-local
               model_path, tool_parser, serving_wait_timeout,
               reset, sample, debug, only_exact,
               gcloud_project_id, gcloud_location,
               tools_type,
               is_batch):
    eval_type = inspect.stack()[0][3]
    run_evaluate(
      eval_type, f'FunctionChat-{eval_type.capitalize()}',
      model,
      input_path,
      temperature, api_key,
      base_url, served_model_name, 
      model_path, tool_parser, serving_wait_timeout,
      reset, sample, debug, only_exact,
      gcloud_project_id, gcloud_location,
      system_prompt_path=system_prompt_path,
      tools_type=tools_type,
      is_batch=is_batch
    )

@cli.command()
@default_eval_options
def common(model, input_path,
           # common
           temperature, api_key,
           # inhouse
           base_url, served_model_name, 
           # inhouse-local
           model_path, tool_parser, serving_wait_timeout,
           # eval option
           reset, sample, debug, only_exact,
           # gemini option
           gcloud_project_id, gcloud_location,
           is_batch):

    eval_type = inspect.stack()[0][3]
    run_evaluate(
      eval_type, os.path.splitext(os.path.basename(input_path))[0],
      model,
      input_path,
      temperature, api_key,
      base_url, served_model_name, 
      model_path, tool_parser, serving_wait_timeout,
      reset, sample, debug, only_exact,
      gcloud_project_id, gcloud_location,
      is_batch=is_batch
    )


if __name__ == '__main__':
    cli()
