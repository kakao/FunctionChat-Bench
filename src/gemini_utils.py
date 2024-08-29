#!/usr/bin/env python3
import sys
import json
import vertexai
from vertexai import generative_models
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)
import google.api_core


def convert_messages_gemini(messages):
    '''
    [
        Content(role="user", parts=[
            Part.from_text(prompt + """Give your answer in steps with lots of detail
                and context, including the exchange rate and date."""),
        ]),
        Content(role="function", parts=[
            Part.from_dict({
                "function_call": {
                    "name": "get_exchange_rate",
                }
            })
        ]),
        Content(role="function", parts=[
            Part.from_function_response(
                name="get_exchange_rate",
                response={
                    "content": api_response.text,
                }
            )
        ])
    ]
    '''
    gemini_system_instruction, gemini_messages = [], []
    for idx, message in enumerate(messages):
        if message['role'] == 'system':
            gemini_system_instruction.append(message['content'])
        elif message['role'] == 'user':
            gemini_messages.append(Content(role="user", parts=[Part.from_text(message['content']),]))
        elif message['role'] == 'assistant':
            parts = []
            if 'content' in message and message['content'] is not None:
                parts.append(Part.from_text(message['content']))
            if 'tool_calls' in message and message['tool_calls'] is not None:
                for tool_call in message['tool_calls']:
                    parts.append(Part.from_dict({"function_call": {"name": tool_call["function"]["name"], "args": json.loads(tool_call["function"]["arguments"])}}))
            if len(parts) > 0:
                gemini_messages.append(Content(role="model", parts=parts))
        elif message['role'] == 'tool':
            func_name = messages[idx - 1]['tool_calls'][0]['function']['name']
            gemini_messages.append(Content(role="function", parts=[Part.from_function_response(name=func_name, response={"content": message['content']})]))
    return gemini_system_instruction, gemini_messages


def convert_tools_gemini(tools):
    '''
    get_current_weather_func = FunctionDeclaration(
        name="get_current_weather",
        description="Get the current weather in a given location",
        # Function parameters are specified in OpenAPI JSON schema format
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"}},
        },
    )

    weather_tool = Tool(
        function_declarations=[get_current_weather_func],
    )

    '''
    gemini_tools = []
    for idx, tool in enumerate(tools):
        func = tool["function"]
        func_name = func["name"]
        func_desc = func["description"]
        func_param = func["parameters"] if len(func["parameters"]) > 0 else {"type": "object", "properties": None}

        gemini_tools.append(FunctionDeclaration(name=func_name, description=func_desc, parameters=func_param))
    return Tool(function_declarations=gemini_tools)


def convert_gemini_to_response(gemini_response):
    '''
    {'role': 'model', 'parts': [{'function_call': {'name': 'getTodayBoxOfficeRanking', 'args': {}}}]}

    parallel function calling은 미지원
    '''
    response = {"role": "assistant", "content": None, "tool_calls": None}
    tool_calls = []

    for part in gemini_response["parts"]:
        if "text" in part:
            response["content"] = part["text"]
        elif "function_call" in part:
            func_call = part["function_call"]
            tool_calls.append({"type": "function", "function": {"name": func_call["name"], "arguments": json.dumps(func_call["args"], ensure_ascii=False)}})
    if len(tool_calls) > 0:
        response["tool_calls"] = tool_calls

    return response


def call_gemini_model(gemini_model, gemini_temperature, gemini_system_instruction, gemini_tools, gemini_messages):

    gemini_model = GenerativeModel(
        model_name=gemini_model,
        generation_config={"temperature": gemini_temperature},
        system_instruction=gemini_system_instruction,
        tools=[gemini_tools])

    # Safety config
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-attributes?hl=ko&cloudshell=false
    safety_config = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }

    try:
        response = gemini_model.generate_content(gemini_messages, safety_settings=safety_config)
        response = response.to_dict()
    except google.api_core.exceptions.InternalServerError as e:
        print(f'{e}, {gemini_messages}')
        response = {"candidates": [{"finish_reason": "ERROR", "content": {"role": "model", "parts": [{"text": None}]}}]}
    return response


if __name__ == '__main__':
    vertexai.init(project='mm-agent-416400', location='asia-northeast3')

    fp = open("gemini_call_log.txt", "w")

    gemini_model = 'gemini-1.5-pro-preview-0514'
    # gemini_model = 'gemini-1.5-flash-preview-0514'
    # gemini_model = 'gemini-1.0-pro-002'

    for buf in sys.stdin:
        api_request = json.loads(buf)  # read openai request format
        gemini_temperature = api_request['temperature'] if 'temperature' in api_request else 0.1
        gemini_system_instruction, gemini_messages = convert_messages_gemini(api_request['messages'])
        gemini_tools = convert_tools_gemini(api_request['tools'])

        gemini_messages_dumps = json.dumps([msg.to_dict() for msg in gemini_messages], ensure_ascii=False)
        print('=== REQUEST')
        print(f'gemini_model = {gemini_model}')
        print(f'gemini_system_instruction = {json.dumps(gemini_system_instruction, ensure_ascii=False)}')
        print(f'gemini_messages = {gemini_messages_dumps}')
        print(f'gemini_tools = {json.dumps(gemini_tools.to_dict(), ensure_ascii=False)}\n')
        print('=== REQUEST', file=fp)
        print(f'gemini_model = {gemini_model}', file=fp)
        print(f'gemini_system_instruction = {json.dumps(gemini_system_instruction, ensure_ascii=False)}', file=fp)
        print(f'gemini_messages = {gemini_messages_dumps}', file=fp)
        print(f'gemini_tools = {json.dumps(gemini_tools.to_dict(), ensure_ascii=False)}\n', file=fp)

        response = call_gemini_model(gemini_model=gemini_model,
                                     gemini_temperature=gemini_temperature,
                                     gemini_system_instruction=gemini_system_instruction,
                                     gemini_tools=gemini_tools, gemini_messages=gemini_messages)
        print('=== RESPONSE')
        print(json.dumps(response, ensure_ascii=False, indent=2))

    fp.close()
