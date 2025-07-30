import os
import json
import time
import requests
import subprocess
import pandas as pd
from tqdm import tqdm
"""
This is a package that collects commonly used basic utilities.
"""


def is_exist_file(file_path):
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            return False
        else:
            return True
    return False


def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except PermissionError:
        print(f"Permission denied: cannot delete {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {path}: {e}")


def load_to_tsv(file_name):
    df = pd.read_csv(file_name, delimiter='\t', lineterminator='\n')
    return [item for idx, item in df.iterrows()]


def load_to_json(file_name):
    datas = None
    with open(file_name, 'r') as ff:
        try:
            datas = json.loads(ff.read())
        except Exception as e:
            print(file_name)
            print("[Exception]", e)
            raise e
    return datas


def load_to_jsonl(input_file_path):
    output = []
    with open(input_file_path) as f:
        for line in tqdm(f.readlines()):
            try:
                output.append(json.loads(line))
            except Exception as e:
                print(line)
                print("[Exception]", e)
                raise e
    return output


def save_to_jsonl(data, filename):
    if isinstance(data, list):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        raise Exception(f"save_to_jsonl error : data type is invalid. ({type(data)})")


def get_output_filename(source_file):
    if '/' in source_file:
        filepath, filename = source_file.rsplit('/', 1)
    else:
        filepath, filename = '.', source_file
    filename_prefix = filename.rsplit('.', 1)[0]
    return f"{filepath}/{filename_prefix}.convert.jsonl"


def save_cache(data, cache_path):
    with open(cache_path, 'w') as f:
        f.write(json.dumps(data, ensure_ascii=False))
    return cache_path


def load_cache(cache_path):
    with open(cache_path, 'r') as f:
        return json.loads(f.read())


def create_directory_if_not_exists(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully or already exists.")
    except Exception as e:
        print(f"Failed to create directory '{directory_path}'. Error: {e}")


def wait_for_server(url, timeout=500):
    start_time = time.time()
    with tqdm(total=timeout, desc="Waiting") as pbar:
        while time.time() - start_time < timeout:
            pbar.update(10)
            try:
                response = requests.get(url)  # noqa: E501
                if response.status_code == 200:
                    print("<<< Server is ready! >>>")
                    return True
            except Exception:
                # print("retry", e)
                pass
            time.sleep(10)
    raise Exception("Server did not start within the timeout period.")


def compare_file_line_counts(file_a, file_b):
    try:
        with open(file_a, 'r') as fa, open(file_b, 'r') as fb:
            lines_a = sum(1 for _ in fa)
            lines_b = sum(1 for _ in fb)
            return lines_a == lines_b
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def convert_tools_alphachat(tools):
    '''
    Modify the tools format for LiteLLM
    '''
    gemini_tools = []
    for idx, tool in enumerate(tools):
        if tool['function']['parameters'] == {}:
            tool['function']['parameters'] = { "type": "object", "properties": {}, "required": [] }
        gemini_tools.append(tool)
    return gemini_tools


def get_git_info():
    def run_git_command(cmd):
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return result.stdout.strip()
    branch = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_git_command(["git", "rev-parse", "HEAD"])[:7]  # 짧은 커밋 해시
    tag = run_git_command(["git", "describe", "--tags", "--abbrev=0"])

    if branch.startswith("tags/"):
        return branch.replace("tags/", "")
    elif branch == "HEAD":
        # detached 상태
        return [tag, f"deteched#{commit}"]
    else:
        return [tag, f"{branch}#{commit}"]
