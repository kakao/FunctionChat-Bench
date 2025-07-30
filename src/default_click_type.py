import click
"""
This package is a collection of argument classes designed to support optional arguments using Click.
"""


# default set local api
DEFAULTS = {
    "api_key": "inhouse",
    "base_url": "http://localhost/v1",
    "reset": False,
    "sample": False,
    "debug": False,
    "only_exact": False,
    "is_batch": True,
    # gemini
    "gcloud_project_id": "user_gcloud_project_id",
    "gcloud_location": "user_gcloud_project_location",
    # for local_inference 
    "tool_parser": "functionary_llama_v3",
    "model_name": "model_name",
    "model_path": "/home/inhouse/train_model/model_name",
    "served_model_name": "served_model_name",
    "serving_wait_timeout": 500,
}


class DefaultBaseUrlPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['base_url']
        return super().prompt_for_value(ctx)

class DefaultServedModelNamePromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['served_model_name']
        return super().prompt_for_value(ctx)

class DefaultModelPathPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['model_path']
        return super().prompt_for_value(ctx)

class DefaultServingWaitTimeoutPromptOptions(click.Option):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('type', int)
        super().__init__(*args, **kwargs)

    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['serving_wait_timeout']
        return super().prompt_for_value(ctx)

class DefaultResetPromptOptions(click.Option):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('type', click.BOOL)
        super(DefaultResetPromptOptions, self).__init__(*args, **kwargs)

    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['reset']
        return super().prompt_for_value(ctx)


class DefaultSamplePromptOptions(click.Option):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('type', click.BOOL)
        super(DefaultSamplePromptOptions, self).__init__(*args, **kwargs)

    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['sample']
        return super().prompt_for_value(ctx)


class DefaultDebugPromptOptions(click.Option):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('type', click.BOOL)
        super(DefaultDebugPromptOptions, self).__init__(*args, **kwargs)

    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['debug']
        return super().prompt_for_value(ctx)

class DefaultBatchPromptOptions(click.Option):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('type', click.BOOL)
        super(DefaultBatchPromptOptions, self).__init__(*args, **kwargs)

    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['is_batch']
        return super().prompt_for_value(ctx)

class DefaultOnlyExactPromptOptions(click.Option):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('type', click.BOOL)
        super(DefaultOnlyExactPromptOptions, self).__init__(*args, **kwargs)

    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['only_exact']
        return super().prompt_for_value(ctx)

class DefaultGPidPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['gcloud_project_id']
        return super().prompt_for_value(ctx)


class DefaultGLocPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['gcloud_location']
        return super().prompt_for_value(ctx)


class DefaultApiKeyPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['api_key']
        return super().prompt_for_value(ctx)

class DefaultToolParserPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['tool_parser']
        return super().prompt_for_value(ctx)
