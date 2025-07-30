import time
import openai
import traceback
from functools import wraps


def get_openai_batch_format(custom_id, openai_model, messages, max_tokens=8192):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": openai_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
    }
