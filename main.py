from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
import httpx
import sys

import threading
import os
import time


def log_before_retry(retry_state):
    """Callback function to log a message before each retry."""
    # The outcome attribute contains details about the last attempt's result or exception
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        print(
            f"Retrying function '{retry_state.fn.__name__}': "
            f"Attempt {retry_state.attempt_number} failed with {exception}. "
            f"Waiting {retry_state.upcoming_sleep:.1f} seconds before next attempt.",
            end="\r",
        )


@retry(stop=stop_after_attempt(100), wait=wait_fixed(10), before_sleep=log_before_retry)
def check_connection(base_url):
    with httpx.Client() as client:
        response = client.get(base_url)
    return True


def call_model(client, model: str, prompt: str) -> str:
    """Returns the model's text output.

    Tries Responses API first, then falls back to Chat Completions.
    """
    start_time = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            extra_body={"thinking": {"type": "enabled"}},
            stream_options={"include_usage": True},
        )
        total_tokens = 0
        for chunk in stream:
            if not chunk.choices:
                break
            # Print the content incrementally as it arrives
            print(chunk.choices[0].delta.content or "", end="")
            if chunk.choices[0].delta.content:
                total_tokens += len(chunk.choices[0].delta.content)

        end_time = time.perf_counter()
        print(f"\n\n--------\n\nTotal return tokens {total_tokens}")
        print(f"Elapsed time: {end_time - start_time:.4f} seconds")
        print(f"{total_tokens / (end_time - start_time):.4f} Tokens/Second")
        return total_tokens
    except Exception as e_chat:
        raise RuntimeError(f"Chat Completions failed: {e_chat}")


def main(head_node):
    base_url = f"http://{head_node}:8000"
    api_key = os.getenv("TOKEN", "")
    try:
        check_connection(base_url=f"{base_url}/health")
    except KeyboardInterrupt:
        print("Stopping tests")
        exit(0)

    client = OpenAI(api_key=api_key, base_url=f"{base_url}/v1")
    models = client.models.list().data
    for model in models:
        print(f"Testing {model.id} waiting to start")
        prompt = "Hello! Please write me a python program to make a FastAPI application for data transfers. Include how to initialize the project using uv and models using pydantic."
        num_threads = 1
        threads = {}
        for i in range(num_threads):
            try:
                threads[i] = threading.Thread(
                    target=call_model,
                    args=(
                        client,
                        model.id,
                        prompt,
                    ),
                )
                threads[i].start()
                # prompt = input("\n\n>> ")
            except KeyboardInterrupt:
                print("Stopping tests")
                break

        [threads[i].join() for i in range(num_threads)]

        print("\n\n--------\n\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Need node as first argument")
