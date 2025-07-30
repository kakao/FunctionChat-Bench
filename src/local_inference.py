from src import utils
from types import SimpleNamespace


def initialize_vllm(model_path, model_name, tool_parser, serving_wait_timeout):
    script_directory = 'kanana_trainer/kanana_trainer/inference'
    import subprocess

    command = [
        "python3", "vllm_fc.py",
        "--model", model_path,
        "--served-model-name", model_name,
        "--tool-call-parser", tool_parser,
        "--port", "8000",
        "--enable-auto-tool-choice"
    ]
    log_path = "./logs"
    utils.create_directory(log_path)
    stdout_file = open(f'{log_path}/{model_name}.stdout.log', 'a')
    stderr_file = open(f'{log_path}/{model_name}.stderr.log', 'a')

    print("Starting VLLM load")
    process = subprocess.Popen(
        command,
        stdout=stdout_file,
        stderr=stderr_file,
        cwd=script_directory)
    process_meta = SimpleNamespace()
    process_meta.process = process
    process_meta.stdout_file = stdout_file
    process_meta.stderr_file = stderr_file
    if process.returncode == None:
        if utils.wait_for_server("http://0.0.0.0:8000/v1/models", serving_wait_timeout):
            print("Loaded VLLM model")
            print(f"Model path: {model_path}")
    else:
        stdout, stderr = process.communicate()
        print("failed VLLM load")
        print(f"Error detected with return code {process.returncode}:\n{stderr.decode('utf-8')}")
        kill_vllm(process_meta)
        return None
    return process_meta


def kill_vllm(process_meta):
    process = process_meta.process
    stdout_file = process_meta.stdout_file
    stderr_file = process_meta.stderr_file
    if process is not None:
       process.terminate()
       return_code = process.wait()
       if return_code == 0:
           print("Closed VLLM model")
       else:
           try:
               os.kill(process.pid, 0)
           except:
               print(f"failed process kill {self.process.pid}")
           else:
               print(f"forced Kill VLLM model {return_code}")
