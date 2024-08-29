#!/usr/bin/env python3

import os
import click
import inspect

from src import utils
from src.default_click_type import (
    DefaultBaseUrlPromptOptions,
    DefaultModelPathPromptOptions,
    DefaultResetPromptOptions,
    DefaultSamplePromptOptions,
    DefaultDebugPromptOptions,
    DefaultGPidPromptOptions,
    DefaultGLocPromptOptions,
    DefaultApiKeyPromptOptions
)

from src.payload_creator import PayloadCreatorFactory
from src.response_handler import ResponseHandler
from src.evaluation_handler import EvaluationHandler


REPO_NAME = "FunctionChat-Bench"
CUR_PATH = os.path.abspath(__file__)
REPO_PATH = f"{CUR_PATH.split(REPO_NAME)[0]}{REPO_NAME}"


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
    f = click.option('--system_prompt_path', prompt='system_prompt_path', help='system prompt file path')(f)
    # test option
    f = click.option('--reset', prompt='recreate request file', help='reset request file', cls=DefaultResetPromptOptions)(f)
    f = click.option('--sample', prompt='Run only 1 case.', help='run sample', cls=DefaultSamplePromptOptions)(f)
    f = click.option('--debug', prompt='debug flag', help='debugging', cls=DefaultDebugPromptOptions)(f)
    # openai type
    f = click.option('--temperature', prompt='temperature', help='generate temperature', default=0.1)(f)
    f = click.option('--api_key', prompt='model api key', help='api key', cls=DefaultApiKeyPromptOptions)(f)
    f = click.option('--base_url', prompt='model api url', help='base url', cls=DefaultBaseUrlPromptOptions)(f)
    # openai - hosting server type
    f = click.option('--model_path', prompt='inhouse model path', help='model path in header', cls=DefaultModelPathPromptOptions)(f)
    # gemini
    f = click.option('--gcloud_project_id', prompt='gemini project id', help='google pid', cls=DefaultGPidPromptOptions)(f)
    f = click.option('--gcloud_location', prompt='gemini location', help='google cloud location', cls=DefaultGLocPromptOptions)(f)
    return f


def singlecall_eval_options(f):
    f = click.option('--tools_type', prompt='tools type', help='tools_type = {exact, 4_random, 4_close, 8_random, 8_close}')(f)
    return f


# program command
@cli.command()
@default_eval_options
def dialog(model,
           input_path, system_prompt_path,
           temperature, api_key, base_url, model_path,
           reset, sample, debug,
           gcloud_project_id, gcloud_location):
    eval_type = inspect.stack()[0][3]
    TEST_PREFIX = f'FunctionChat-{eval_type.capitalize()}'

    print(f"[[{model} {TEST_PREFIX} evaluate start]]")
    utils.create_directory(f'{REPO_PATH}/output/')

    request_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.input.jsonl'
    predict_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.{model}.output.jsonl'
    eval_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.{model}.eval.jsonl'
    eval_log_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.{model}.eval_report.tsv'

    api_request_list = PayloadCreatorFactory.get_payload_creator(
        eval_type, temperature, system_prompt_path
    ).create_payload(
        input_file_path=input_path, request_file_path=request_file_path, reset=reset)
    api_response_list = ResponseHandler(
        model, api_key, base_url, model_path, gcloud_project_id, gcloud_location
    ).fetch_and_save(
        api_request_list, predict_file_path, reset, sample, debug
    )
    EvaluationHandler(eval_type).evaluate(
        api_request_list, api_response_list,
        eval_file_path, eval_log_file_path,
        reset, sample, debug
    )


@cli.command()
@default_eval_options
@singlecall_eval_options
def singlecall(model,
               input_path, tools_type,
               system_prompt_path,
               temperature, api_key, base_url, model_path,
               reset, sample, debug,
               gcloud_project_id, gcloud_location):

    eval_type = inspect.stack()[0][3]
    TEST_PREFIX = f'FunctionChat-{eval_type.capitalize()}'

    print(f"[[{model} {TEST_PREFIX} {tools_type} evaluate start]]")
    utils.create_directory(f'{REPO_PATH}/output/')

    request_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.input.jsonl'
    predict_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.{model}.{tools_type}.output.jsonl'
    eval_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.{model}.{tools_type}.eval.jsonl'
    eval_log_file_path = f'{REPO_PATH}/output/{TEST_PREFIX}.{model}.{tools_type}.eval_report.tsv'

    api_request_list = PayloadCreatorFactory.get_payload_creator(
        eval_type, temperature, system_prompt_path
    ).create_payload(
        input_file_path=input_path, request_file_path=request_file_path,
        reset=reset, tools_type=tools_type
    )
    api_response_list = ResponseHandler(
        model, api_key, base_url, model_path,
        gcloud_project_id, gcloud_location
    ).fetch_and_save(
        api_request_list, predict_file_path, reset, sample, debug
    )
    EvaluationHandler(eval_type).evaluate(
        api_request_list, api_response_list,
        eval_file_path, eval_log_file_path,
        reset, sample, debug
    )


if __name__ == '__main__':
    cli()
