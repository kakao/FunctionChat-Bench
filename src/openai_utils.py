import time
import openai
from functools import wraps


def retry_on_limit(func, retries=5, wait=120):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError as error:
                print(str(error))
                time.sleep(wait)
        raise openai.RateLimitError
    return wrapper
