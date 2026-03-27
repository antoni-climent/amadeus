import unittest
import json
from pathlib import Path

from backend.inference import ModelService


class DummyInputs(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class DummyTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values[0]))

    def __getitem__(self, index):
        return self.values[index]


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.last_messages = messages
        return "prompt"

    def __call__(self, text, return_tensors):
        return DummyInputs(DummyTensor([[10, 11, 12]]))

    def decode(self, _response_ids, skip_special_tokens):
        return "Answer"


class DummyModel:
    def generate(self, **kwargs):
        return [[10, 11, 12, 13, 14]]


class ModelServiceTests(unittest.TestCase):
    def test_get_system_prompt_returns_empty_string_when_missing(self):
        service = ModelService(system_prompt_path=Path("/tmp/does-not-exist.txt"))
        self.assertEqual(service.get_system_prompt(), "")

    def test_generate_returns_history_without_system_message(self):
        service = ModelService()
        service.model = DummyModel()
        service.tokenizer = DummyTokenizer()

        result = service.generate(
            message="What did I say?",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            system_prompt="You are Kurisu.",
            max_new_tokens=32,
        )

        self.assertEqual(
            service.tokenizer.last_messages,
            [
                {"role": "system", "content": "You are Kurisu."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What did I say?"},
            ],
        )
        self.assertEqual(
            result["history"],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What did I say?"},
                {"role": "assistant", "content": "Answer"},
            ],
        )

    def test_format_sse_event_uses_expected_shape(self):
        service = ModelService()

        event = service.format_sse_event("delta", {"delta": "Hi", "response": "Hi"})

        self.assertEqual(
            event,
            f"event: delta\ndata: {json.dumps({'delta': 'Hi', 'response': 'Hi'})}\n\n",
        )


if __name__ == "__main__":
    unittest.main()
