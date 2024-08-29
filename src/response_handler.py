import json
from tqdm import tqdm
from src import utils
from src.api_executor import APIExecutorFactory


class ResponseHandler:
    """
    A class responsible for managing API responses, including loading cached responses.
    """
    def __init__(self, model, api_key, base_url, model_path, gcloud_project_id, gcloud_location):
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
                                                           base_url=base_url, model_path=model_path,
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

    def fetch_and_save(self, api_request_list, predict_file_path, reset, sample, debug):
        """
        Fetches responses from the API and saves them. If responses are partially cached, it continues from where it left off.

        Parameters:
            api_request_list (list): List of API requests to process.
            predict_file_path (str): File path to save the responses.
            reset (bool): If True, it overwrite existing cached responses; if False, append to them.
            sample (bool): If True, it executes only a single input to fetch the response. (e.g., for quick testing).
            debug (bool): If True, it print detailed debug information.

        Returns:
            list: A list of all responses fetched and saved.
        """
        outputs = []
        # 1. check continuos
        if reset is False:
            outputs = self.load_cached_response(predict_file_path, len(api_request_list))
            if len(outputs) == len(api_request_list):
                return outputs
        write_option = 'a' if reset is False else 'w'
        start_index = len(outputs)
        # 2. fetch
        print(f" ** start index : {start_index} ..(reset is {reset})")
        fp = open(predict_file_path, write_option)
        for idx, api_request in enumerate(tqdm(api_request_list[start_index:])):
            if sample is True and idx == 1:
                break
            response_output = self.executor.predict(api_request)
            outputs.append(response_output)
            # 3. save
            fp.write(f'{json.dumps(response_output, ensure_ascii=False)}\n')
        fp.close()
        print(f"[[model response file : {predict_file_path}]]")
        return outputs
