#!/usr/bin/env python3

import click
import json
from pathlib import Path


def load_jsonl(file):
    return [json.loads(line) for line in open(file)]


def get_file_names(path: Path, suffix: str = ".eval.jsonl"):
    return sorted([file.name for file in path.glob(f"*{suffix}")])


def get_unique(doc):
    report = doc["report_arguments"]
    return (int(report["serial_num"]), report.get("category", ""))


def pass_or_fail(doc):
    return doc["report_arguments"]["is_pass"] if doc else None


def eval_step(doc):
    if doc is None:
        return None
    return "exact" if "exact" in doc["evaluate_response"] else "rubric"


def count_pass_or_fail(datum):
    result = {"pass": 0, "fail": 0, "err": 0}
    for data in datum.values():
        res = pass_or_fail(data)
        if res in result:
            result[res] += 1
        else:
            result["err"] += 1
    return result


def diff_datum(b_datum, a_datum):
    diffs = {}
    unique_keys = [get_unique(v) for v in b_datum.values()] + [get_unique(v) for v in a_datum.values()]
    for unique_key in sorted(list(set(unique_keys))):
        b_data = b_datum.get(unique_key)
        a_data = a_datum.get(unique_key)

        diffs[unique_key] = {
            "b": {"pass": pass_or_fail(b_data), "step": eval_step(b_data)},
            "a": {"pass": pass_or_fail(a_data), "step": eval_step(a_data)}
        }
        if diffs[unique_key]["b"] == diffs[unique_key]["a"]:
            diffs.pop(unique_key)

    return diffs


def make_abst(diff):
    pass_changed = f"{diff['b']['pass']} -> {diff['a']['pass']}"
    b_step = diff['b']['step']
    a_step = diff['a']['step']
    if b_step != a_step:
        eval_changed = f"{b_step} -> {a_step}"
    elif b_step == "exact":
        eval_changed = f"exact eval changed"
    else:
        eval_changed = f"judge changed"
    return f"{pass_changed:20s}\t{eval_changed:20s}"


def gather_scores(output_models: list[Path]) -> (list[str], dict):
    # count pass of fail and get unique names of evaluation file
    eval_names = set()
    scores = {}
    for output_model in output_models:
        scores[output_model.name] = {}
        for input_file in get_file_names(output_model, ".input.jsonl"):
            # ex) test.sequential-basic.input.jsonl -> test.sequential-basic.*.eval.jsonl
            eval_file = input_file.replace('.input', '.*.eval')
            # ex) test.sequential-basic.input.jsonl -> sequential-basic
            eval_name = input_file.replace(".input.jsonl", "").replace("test.", "").replace("FunctionChat-", "")
            try:
                file_path = next(output_model.glob(eval_file))
                datum = {get_unique(doc): doc for doc in load_jsonl(file_path)}
                scores[output_model.name][eval_name] = count_pass_or_fail(datum)
                eval_names.add(eval_name)
            except StopIteration:
                print(f"cannot find {output_model}/{eval_file}")
                continue
    return eval_names, scores


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model-name", "-m", required=True)
@click.option("--before", "-b", required=True)
@click.option("--after", "-a", required=True)
@click.option("--file", "-f")
@click.option("--show", "-s", "fields", multiple=True)
def ab(model_name: str, before: str, after: str, file: str = "", fields: list[str] = []):
    home_path = Path(__file__).parent.parent.resolve()
    b_path = home_path / "output" / f"{model_name}.{before}"
    a_path = home_path / "output" / f"{model_name}.{after}"
    for file_name in set(get_file_names(b_path) + get_file_names(a_path)):
        if file and file not in file_name:
            continue
        b_file = b_path / file_name
        if not b_file.exists():
            print(f"{file_name} not exists in {b_path}")
            continue
        a_file = a_path / file_name
        if not a_file.exists():
            print(f"{file_name} not exists in {a_path}")
            continue
        b_datum = {get_unique(doc): doc for doc in load_jsonl(b_file)}
        a_datum = {get_unique(doc): doc for doc in load_jsonl(a_file)}
        if len(b_datum) != len(a_datum):
            print(f"length of datum mismatch {len(b_datum)=} {len(a_datum)=}")
            continue
        diffs = diff_datum(b_datum, a_datum)
        print(f"checking {file_name}")
        for key, diff in diffs.items():
            b_data = b_datum.get(key)
            a_data = a_datum.get(key)
            # print(f"##### {str(key):30s} {make_abst(diff)} #####")
            print(f"{str(key):30s}\t{make_abst(diff)}")
            if fields:
                print("<" * 100)
            for field in fields:
                print(json.dumps(b_data[field], indent=2, ensure_ascii=False) if b_data else "None")
            if fields:
                print(">" * 100)
            for field in fields:
                print(json.dumps(a_data[field], indent=2, ensure_ascii=False) if a_data else "None")

    eval_names, scores = gather_scores([b_path, a_path])
    b_key = b_path.name
    a_key = a_path.name
    print(f"\n{'eval_name':30s} {before:>20s} {after:>20s}")
    print(f"{'':30s} {'pass fail  err':>20s} {'pass fail  err':>20s}")
    for eval_name in sorted(eval_names):
        b_score = scores.get(b_key, {}).get(eval_name, {"pass": 0, "fail": 0, "err": 0})
        a_score = scores.get(a_key, {}).get(eval_name, {"pass": 0, "fail": 0, "err": 0})
        b_score_str = f"{b_score['pass']:4d} {b_score['fail']:4d} {b_score['err']:4d}"
        a_score_str = f"{a_score['pass']:4d} {a_score['fail']:4d} {a_score['err']:4d}"
        print(f"{eval_name:30s} {b_score_str:>20s} {a_score_str:>20s}")


@cli.command()
@click.option("--output", "output_path", type=Path)
@click.option("--count",is_flag=True, default=False)
@click.option("--raw", is_flag=True, default=False)
def score(output_path: Path = None, raw: bool = False, count: bool = False):
    def shorten(name: str):
        model_name = name[:45]
        model_name += " ..." if len(name) > 50 else ""
        return model_name

    if output_path is None:
        home_path = Path(__file__).parent.parent.resolve()
        output_path = home_path / "output"
    output_models = [name for name in output_path.iterdir() if name.is_dir()]

    eval_names, scores = gather_scores(output_models)

    if raw:
        print(json.dumps(scores, indent=2))

    from rich.console import Console
    from rich.table import Table

    console = Console()
    layout_table = Table(show_lines=False)
    # header
    layout_table.add_column("model")
    for eval_name in sorted(eval_names):
        layout_table.add_column(eval_name[:30])

    model_names = sorted(scores)
    model_name_table = Table(show_header=False, show_lines=False, show_edge=False)
    for model_name in model_names:
        model_name_table.add_row(shorten(model_name))

    big_row = [model_name_table]
    # body
    for eval_name in sorted(eval_names):
        # nested table
        eval_table = Table(show_header=False, show_lines=False, show_edge=False)
        for output_model in model_names:
            try:
                total = sum(scores[output_model][eval_name].values())
                pass_count = scores[output_model][eval_name]['pass']
                ratio = pass_count / total
            except KeyError:
                total, pass_count, ratio = None, None, None

            eval_row = []
            if count:
                eval_row.append(f"{pass_count}" if pass_count else "-")
            eval_row.append(f"{ratio:.4f}" if ratio else "-")
            # add row to nested table
            eval_table.add_row(*eval_row)
        big_row.append(eval_table)
    layout_table.add_row(*big_row)
    console.print(layout_table)


if __name__ == "__main__":
    cli()
