# Terra Effort Sweep

Compare reasoning-effort settings for **GPT-6.1 Terra** using your own evaluation tasks.

The script measures:

- Exact-match accuracy
- Average request latency
- Successful and failed requests
- Estimated token cost when usage is available

It writes per-request results, a summary CSV, and a comparison chart.

[View GPT-6.1 Terra on CometAPI](https://www.cometapi.com/models/openai/gpt-6-1-terra/?utm_source=github&utm_medium=distribution&utm_campaign=bom&utm_content=gpt-6-1-terra)

> The API integration has not been tested against the live CometAPI endpoint. Confirm the model ID, supported reasoning settings, and pricing before running. Settings listed here are candidates to test, not independently verified model specifications.

## Requirements

- Python 3.10 or later
- A CometAPI API key
- Access to the model through a Responses-compatible endpoint

## Setup

Install dependencies:

```bash
python -m pip install --upgrade openai matplotlib
```

Set your API key:

```bash
export COMETAPI_KEY="your_key_here"
```

Do not commit API keys or confidential evaluation data.

## Create your tasks

Save one JSON object per line in `tasks.jsonl`:

```jsonl
{"prompt": "What is 17% of 2,340? Answer with the number only.", "answer": "397.8"}
{"prompt": "Which is larger, 0.9 or 0.11? Answer with the number only.", "answer": "0.9"}
```

Both `prompt` and `answer` must be non-empty strings.

The default grader ignores capitalization and extra whitespace, then requires an exact match. It does not use substring matching.

For example, `not 397.8` will not pass as `397.8`.

Equivalent numeric formats such as `397.80` and `397.8` are treated as different answers. Use a task-specific grader if that is inappropriate.

## Run

Test all six candidate settings:

```bash
python sweep.py tasks.jsonl --repeats 3
```

Or select settings supported by your endpoint:

```bash
python sweep.py tasks.jsonl \
  --levels low medium high \
  --repeats 3
```

Supply current token prices for cost estimates:

```bash
python sweep.py tasks.jsonl \
  --input-price 0.80 \
  --output-price 4.00
```

Prices are in dollars per million tokens. These example rates are not verified current prices.

Outputs:

- `requests.csv`: individual results, responses, and errors
- `results.csv`: summary by effort level
- `sweep.png`: accuracy and latency chart

Existing output files are overwritten.

## Script: `sweep.py`

```python
import argparse
import csv
import json
import os
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

LEVELS = ["none", "low", "medium", "high", "xhigh", "max"]


def normalize(text):
    return " ".join(text.split()).casefold()


def load_tasks(path):
    tasks = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number}."
            ) from exc

        if not isinstance(task, dict):
            raise ValueError(
                f"Line {line_number} must contain an object."
            )

        for field in ("prompt", "answer"):
            value = task.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Line {line_number}: {field} "
                    "must be a non-empty string."
                )

        tasks.append(task)

    if not tasks:
        raise ValueError("The task file is empty.")

    return tasks


def write_csv(path, rows):
    with Path(path).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate(client, model, level, task, prices):
    row = {
        "effort": level,
        "status": "error",
        "correct": False,
        "seconds": None,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_cost_usd": None,
        "expected": task["answer"],
        "response": "",
        "error": "",
    }

    start = time.perf_counter()

    try:
        response = client.responses.create(
            model=model,
            reasoning={"effort": level},
            input=task["prompt"],
        )
        row["seconds"] = time.perf_counter() - start

        usage = response.usage
        if usage is not None:
            row["input_tokens"] = usage.input_tokens
            row["output_tokens"] = usage.output_tokens

            if (
                prices is not None
                and usage.input_tokens is not None
                and usage.output_tokens is not None
            ):
                input_price, output_price = prices
                row["estimated_cost_usd"] = (
                    usage.input_tokens * input_price
                    + usage.output_tokens * output_price
                ) / 1_000_000

        row["response"] = response.output_text or ""

        if response.status != "completed":
            raise RuntimeError(
                f"Response status: {response.status}"
            )

        if not row["response"].strip():
            raise RuntimeError("No output text returned.")

        row["status"] = "completed"
        row["correct"] = (
            normalize(row["response"])
            == normalize(task["answer"])
        )

    except Exception as exc:
        if row["seconds"] is None:
            row["seconds"] = time.perf_counter() - start
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument(
        "--model", default="gpt-6.1-terra"
    )
    parser.add_argument(
        "--levels", nargs="+", choices=LEVELS,
        default=LEVELS,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--output-price", type=float)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")

    if len(set(args.levels)) != len(args.levels):
        parser.error("Do not repeat effort levels.")

    supplied_prices = (
        args.input_price is not None,
        args.output_price is not None,
    )
    if any(supplied_prices) and not all(supplied_prices):
        parser.error("Supply both token prices or neither.")

    prices = None
    if all(supplied_prices):
        import math

        prices = (args.input_price, args.output_price)
        if any(
            not math.isfinite(price) or price < 0
            for price in prices
        ):
            parser.error("Prices must be finite and non-negative.")

    api_key = os.environ.get("COMETAPI_KEY")
    if not api_key:
        parser.error("Set COMETAPI_KEY first.")

    try:
        tasks = load_tasks(args.tasks)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    jobs = [
        (level, repeat, index, task)
        for level in args.levels
        for repeat in range(1, args.repeats + 1)
        for index, task in enumerate(tasks, start=1)
    ]
    random.Random(42).shuffle(jobs)

    rows = []

    with OpenAI(
        base_url="https://api.cometapi.com/v1",
        api_key=api_key,
        timeout=180.0,
        max_retries=0,
    ) as client:
        for number, (level, repeat, index, task) in enumerate(
            jobs, start=1
        ):
            print(
                f"{number}/{len(jobs)}: {level}, "
                f"task {index}, repetition {repeat}",
                flush=True,
            )

            row = evaluate(
                client, args.model, level, task, prices
            )
            row["task"] = index
            row["repeat"] = repeat
            rows.append(row)

            # Save progress after every request.
            write_csv("requests.csv", rows)

    summaries = []

    for level in args.levels:
        selected = [
            row for row in rows
            if row["effort"] == level
        ]
        completed = [
            row for row in selected
            if row["status"] == "completed"
        ]
        costs = [
            row["estimated_cost_usd"]
            for row in selected
            if row["estimated_cost_usd"] is not None
        ]

        summary = {
            "effort": level,
            "attempted": len(selected),
            "completed": len(completed),
            "failed": len(selected) - len(completed),
            "exact_match_rate": (
                sum(row["correct"] for row in selected)
                / len(selected)
            ),
            "avg_completed_seconds": (
                statistics.mean(
                    row["seconds"] for row in completed
                )
                if completed else None
            ),
            "requests_with_cost_estimate": len(costs),
            "known_estimated_cost_usd": (
                sum(costs) if costs else None
            ),
        }
        summaries.append(summary)
        print(summary)

    write_csv("results.csv", summaries)

    labels = [row["effort"] for row in summaries]
    accuracy = [
        row["exact_match_rate"] for row in summaries
    ]
    latency = [
        row["avg_completed_seconds"]
        if row["avg_completed_seconds"] is not None
        else float("nan")
        for row in summaries
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(labels, accuracy, marker="o")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact-match rate")
    axes[0].set_title("Failures count as incorrect")

    axes[1].plot(labels, latency, marker="o")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Mean latency: completed requests")

    for axis in axes:
        axis.set_xlabel("Reasoning effort")
        axis.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig("sweep.png", dpi=150)
    plt.close(fig)

    print("\nSaved requests.csv, results.csv, and sweep.png.")


if __name__ == "__main__":
    main()
```

## Reading the results

- `exact_match_rate` includes every attempted request.
  Failed or incomplete requests count as incorrect.
- Latency averages include completed requests only.
  Check failure counts alongside latency.
- Cost is estimated only when prices and usage are available.
  A blank cost means unknown, not free.
- Failed requests may still incur charges.
- Automatic SDK retries are disabled.

An unsupported effort level is an integration failure, not evidence
that the model answered the task incorrectly. Resolve support issues
before interpreting accuracy comparisons.

## Cost assumptions

The estimator applies a flat input rate and a flat output rate to
the usage reported by the endpoint.

It does not separately account for:

- Cached-input discounts
- Tool charges
- Special pricing tiers
- Charges missing from response usage

Confirm whether reported output tokens include billable reasoning
usage. Do not add reasoning tokens again if they are already included.

Compare estimates with actual provider billing.

## Evaluation limitations

Exact matching works best for short, objectively gradable answers.

Use a different evaluator for:

- Numeric answers with equivalent formatting
- Structured extraction
- Summaries and factual reports
- Code generation
- Open-ended writing

This script measures total request time, not time to first token.

Randomized request order reduces simple ordering effects, but does
not control provider load, caching, or network conditions.

Small task sets and repeated runs do not establish broad reliability.
Use representative tasks and a separate holdout set before choosing
production settings.

## Choosing an effort level

Choose based on measured quality, latency, and cost.

Do not assume that every simple task needs `none`, or that `max`
makes consequential work reliable. Customer-facing and financial
workflows still need appropriate validation and review.

## Data handling

`requests.csv` contains expected answers, model responses, and error
messages. Treat it as potentially sensitive.

Only send evaluation data you are authorized to share with CometAPI
and its upstream providers.

Suggested `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
tasks.jsonl
requests.csv
results.csv
sweep.png
```

## Links

- [GPT-6.1 Terra model page and pricing](https://www.cometapi.com/models/openai/gpt-6-1-terra/?utm_source=github&utm_medium=distribution&utm_campaign=bom&utm_content=gpt-6-1-terra)
- [CometAPI documentation](https://apidoc.cometapi.com)

## Disclosure

I share promotional content and links for CometAPI.
The links in this README include tracking parameters.
````
