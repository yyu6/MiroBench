from typing import List, Dict, Tuple, Iterator, Any

import openai
import sys
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai.error import RateLimitError

from src.configs import ModelConfig
from src.prompts import Prompt, Conversation

from .model import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    from token_usage_tracker import record_openai_usage
except Exception:  # pragma: no cover - tracking must never block SynthPAI.
    def record_openai_usage(*args: Any, **kwargs: Any) -> None:
        return


class OpenAIGPT(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        # gpt-5-mini rejects max_tokens and temperature=0
        if "gpt-5" in self.config.name:
            self.config.args.pop("max_tokens", None)
            if "max_completion_tokens" not in self.config.args:
                self.config.args["max_completion_tokens"] = 600
            # gpt-5-mini only supports temperature=1
            self.config.args.pop("temperature", None)
            self.config.args.pop("frequency_penalty", None)
        elif "gemini" in self.config.name.lower():
            # Gemini supports frequencyPenalty in its native API, but its
            # OpenAI-compatible chat endpoint rejects frequency_penalty.
            self.config.args.pop("frequency_penalty", None)
            self.config.args.setdefault("max_tokens", 600)
        elif "max_tokens" not in self.config.args.keys():
            self.config.args["max_tokens"] = 600

    def _predict_call(self, input: List[Dict[str, str]]) -> str:
        input = self._normalize_messages(input)
        if self.config.provider == "azure":
            response = openai.ChatCompletion.create(
                engine=self.config.name, messages=input, **self.config.args
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.config.name, messages=input, **self.config.args
            )
        record_openai_usage(response, model=self.config.name, component="synthpai_openai")
        return response["choices"][0]["message"]["content"]

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        has_content_turn = False
        for message in messages:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})
            if role in {"user", "assistant"}:
                has_content_turn = True

        # Gemini's OpenAI-compatible endpoint rejects system-only chat payloads.
        if normalized and not has_content_turn:
            normalized.append(
                {
                    "role": "user",
                    "content": "Please answer the instruction above.",
                }
            )
        if not normalized:
            normalized.append({"role": "user", "content": "Please continue."})
        return normalized

    def predict(self, input: Prompt, **kwargs) -> str:
        messages: List[Dict[str, str]] = []

        if input.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": input.system_prompt,
                }
            )
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert investigator and detective with years of experience in online profiling and text analysis.",
                }
            ]

        messages += [
            {"role": "user", "content": self.apply_model_template(input.get_prompt())}
        ]

        guess = self._predict_call(messages)

        # response = openai.ChatCompletion.create(
        #     model=self.config.name, messages=messages, **self.config.args
        # )

        # guess = response["choices"][0]["message"]["content"]

        return guess

    def predict_string(self, input: str, **kwargs) -> str:
        input_list = [
            {
                "role": "system",
                "content": "You are an helpful assistant.",
            },
            {"role": "user", "content": input},
        ]

        guess = self._predict_call(input_list)

        # response = openai.ChatCompletion.create(
        #     model=self.config.name, messages=input_list, **self.config.args
        # )

        # guess = response["choices"][0]["message"]["content"]

        return guess

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 4
        base_timeout = kwargs["timeout"] if "timeout" in kwargs else 120

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(inputs)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(inputs):
                # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
                results = executor.map(
                    lambda id: (id, inputs[id], self.predict(inputs[id])),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Profiles",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield (orig, answer)
                        # answered_prompts.append()
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except RateLimitError as r:
                    print(f"Rate_limit {r}")
                    time.sleep(30)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(120, timeout)
                retry_ctr += 1

        # return answered_prompts

    def continue_conversation(self, input: Conversation, **kwargs) -> str:
        input_list: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": input.system_prompt,
            }
        ]

        for message in input.prompts:
            assert message.role is not None
            input_list.append(
                {
                    "role": message.role,
                    "content": message.get_prompt(),  # Simply returns the intermediate text
                }
            )

        guess = None
        while guess is None:
            try:
                guess = self._predict_call(input_list)
            except RateLimitError as r:
                time.sleep(30)
                continue

        return guess
