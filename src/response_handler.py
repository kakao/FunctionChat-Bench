import json
import time
import concurrent
from tqdm import tqdm
import threading

from src import utils
from src.api_executor import APIExecutorFactory


class ResponseHandler:
    """
    A class responsible for managing API responses, including loading cached responses.
    """
    def __init__(self, model, api_key, base_url, served_model_name, gcloud_project_id, gcloud_location):
        """
        Initializes the ResponseHandler with a specific API executor based on the model configuration.

        Parameters:
            model (str): The model identifier used for the API executor.
            api_key (str): API key for authentication with the API service.
            base_url (str): Base URL of the API service.
            model_path (str): Path to the model (if applicable).
            gcloud_project_id (str): Google Cloud project ID (if applicable).
            gcloud_location (str): Location of the Google Cloud project (if applicable).
        """
        self.executor = APIExecutorFactory().get_model_api(model_name=model, api_key=api_key,
                                                           base_url=base_url, served_model_name=served_model_name,
                                                           gcloud_project_id=gcloud_project_id,
                                                           gcloud_location=gcloud_location)

    def load_cached_response(self, predict_file_path, max_size):
        """
        Loads cached responses from a file if available and the number of responses meets the max size.

        Parameters:
            predict_file_path (str): Path to the file containing cached responses.
            max_size (int): Maximum number of responses expected.

        Returns:
            list: A list of cached responses if they exist and meet the expected size; otherwise, an empty list.
        """
        if utils.is_exist_file(predict_file_path):
            outputs = utils.load_to_jsonl(predict_file_path)
            if len(outputs) == max_size:
                print(f"[[already existed response jsonl file]]\npath : {predict_file_path}")
                return outputs
            else:
                print(f"[[continue .. {len(outputs)}/{max_size}]]\n")
                return outputs
        return []

    def fetch_and_save(self, api_request_list, predict_file_path, reset, sample, debug, max_threads=2):
        """
        Fetches responses from the API using multithreading and saves them. If responses are partially cached, it continues from where it left off.

        Parameters:
            api_request_list (list): List of API requests to process.
            predict_file_path (str): File path to save the responses.
            reset (bool): If True, it overwrite existing cached responses; if False, append to them.
            sample (bool): If True, it executes only a single input to fetch the response. (e.g., for quick testing).
            debug (bool): If True, it print detailed debug information.
            max_threads (int): Maximum number of threads to use for API requests.

        Returns:
            list: A list of all responses fetched and saved.
        """
        outputs = []
        try:
            models = self.executor.models()
            print(models)
        except Exception as e:
            print(f"Error fetching models: {e}")
            raise e

        # 1. check existing responses
        if not reset:
            outputs = self.load_cached_response(predict_file_path, len(api_request_list))
            if len(outputs) == len(api_request_list):
                return outputs
        write_option = 'a' if reset is False else 'w'
        outputs = [None] * len(api_request_list) 

        start_time = time.time()
        # 2. fetch responses using multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(self.executor.predict, api_request): idx
                for idx, api_request in enumerate(api_request_list)
            }

            # 3. process completed futures
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            #for future in concurrent.futures.as_completed(futures):
                idx = futures[future] # 병렬처리는 순서가 보장이 안되어서 인덱스 매칭 필요
                response_output = None
                try:
                    # 3번 재시도
                    for _ in range(3):
                        try:
                            response_output = future.result(timeout=30)
                            break
                        except Exception as e:
                            print(f"Error processing request at index {idx}: {e}")
                            time.sleep(1)
                    
                    if response_output is None:
                        response_output = {"role": "assistant", "content": "", "tool_calls": [], "error": f"api response is None"}
                    outputs[idx] = response_output
                except Exception as e:
                    print(f"Error processing request at index {idx}: {e}")
                    response_output = {"role": "assistant", "content": "", "tool_calls": [], "error": str(e)}
                    outputs[idx] = response_output
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time execution: {elapsed_time:.2f} seconds")
        # save response
        with open(predict_file_path, write_option) as fp:
            for response_output in outputs:
                fp.write(f'{json.dumps(response_output, ensure_ascii=False)}\n')
        print(f"[[model response file : {predict_file_path}]]")
        return outputs
