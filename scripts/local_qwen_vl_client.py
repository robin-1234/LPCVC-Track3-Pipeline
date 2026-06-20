#!/usr/bin/env python3
import gc
import json
from pathlib import Path


class LocalVLMError(RuntimeError):
    pass


def _torch_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_model_class():
    errors = []
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText, "AutoModelForImageTextToText"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"AutoModelForImageTextToText: {exc}")
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq, "AutoModelForVision2Seq"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"AutoModelForVision2Seq: {exc}")
    raise LocalVLMError("No supported Qwen-VL AutoModel class available: " + " | ".join(errors))


class LocalQwenVLClient:
    def __init__(self, model_path: str, device_map: str = "auto"):
        import torch
        from transformers import AutoProcessor

        self.model_path = str(Path(model_path).expanduser().resolve())
        config_path = Path(self.model_path) / "config.json"
        if not config_path.exists():
            raise LocalVLMError(f"config.json not found under model_path: {self.model_path}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.architectures = self.config.get("architectures", [])

        model_cls, model_cls_name = _load_model_class()
        self.model_cls_name = model_cls_name
        self.dtype = _torch_dtype()
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self.model.eval()
        except torch.cuda.OutOfMemoryError as exc:
            raise LocalVLMError(f"CUDA OOM while loading model: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LocalVLMError(f"Model loading error with {model_cls_name}: {exc}") from exc

    def _first_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration as exc:
            raise LocalVLMError("Model has no parameters.") from exc

    def _move_inputs(self, inputs):
        device = self._first_device()
        return inputs.to(device)

    def _messages(self, prompt: str, image_path: str | None):
        if image_path:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(Path(image_path).expanduser().resolve())},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def generate(self, prompt: str, image_path: str | None = None, max_new_tokens: int = 500) -> str:
        import torch

        messages = self._messages(prompt, image_path)
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if image_path:
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )
            inputs = self._move_inputs(inputs)
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            input_ids = inputs["input_ids"]
            trimmed = [
                output_ids[len(input_ids_row) :]
                for input_ids_row, output_ids in zip(input_ids, generated)
            ]
            output_text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output_text
        except torch.cuda.OutOfMemoryError as exc:
            raise LocalVLMError(f"CUDA OOM during generation: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LocalVLMError(f"Generation error: {exc}") from exc
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
