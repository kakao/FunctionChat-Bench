import click
"""
This package is a collection of argument classes designed to support optional arguments using Click.
"""


# default set local api
DEFAULTS = {
    "api_key": "inhouse",
    "base_url": "http://localhost/v1",
    "model_path": "/home/inhouse/train_model/model_name",
    "reset": False,
    "sample": False,
    "debug": False,
    # gemini
    "gcloud_project_id": "user_gcloud_project_id",
    "gcloud_location": "user_gcloud_project_location",
}


class DefaultBaseUrlPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['base_url']
        return super().prompt_for_value(ctx)


class DefaultModelPathPromptOptions(click.Option):
    def prompt_for_value(self, ctx):
        q = ctx.obj.get("q")
        if q:
            return DEFAULTS['model_path']
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
