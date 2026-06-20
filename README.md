# LPCVC Track 3: ThinkFake-Style SFT and Qualcomm Deployment Pipeline

## Overview

This repository is a public documentation and source-code repository for an LPCVC Track 3 AI-generated image detection project. The broader workspace retains several related experiments, including SID-Set data construction, Qwen-VL supervised fine-tuning, checkpoint merging, AIMET quantization, ONNX/QNN conversion, Qualcomm validation scripts, and competition packaging work.

The important caveat is that these files belong to multiple project iterations. They should not be read as one fully unified run unless a specific local script, log, config, or report supports that connection.

High-level project narrative:

```text
LPCVC task and sample solution
        |
        v
multimodal real/fake dataset preparation
        |
        v
structured ThinkFake-style supervision
        |
        v
Qwen-VL supervised fine-tuning
        |
        v
checkpoint selection or merge
        |
        v
AIMET quantization
        |
        v
ONNX / QNN conversion
        |
        v
Qualcomm deployment validation
        |
        v
competition submission
```

The public repository documents reproducible source-code workflows. Datasets, model weights, proprietary Qualcomm components, generated deployment binaries, large logs, and competition submission artifacts are not included as public source assets.

## Scope and Publication Boundaries

This README distinguishes documented source workflows from retained local artifacts:

- Source scripts and configuration files describe the SID-Set annotation pipeline and parts of the training/deployment workflow.
- Large model weights, checkpoints, ONNX files, QNN binaries, Qualcomm SDK assets, generated raw inputs, and final submission zips are treated as local retained artifacts.
- Old generated manifests or logs can contain machine-specific absolute paths. Regenerate or normalize them before use on another machine.
- Competition placement is not stated here because the reviewed public files do not include an official leaderboard export, certificate, or final result file.

## Project Iterations

The workspace contains three distinguishable project areas:

1. Competition-era Qwen2-VL deployment pipeline.
   The sample solution documents a Qwen2-VL-2B-Instruct deployment path using AIMET export, ONNX-to-Qualcomm NN preparation, QNN runtime files, Qualcomm AI Hub utilities, and final submission packaging.

2. Later Qwen2.5-VL deployment artifacts.
   Local artifact names and reports refer to a Qwen2.5-VL 3B checkpoint-100 merge/quantization workspace. The checked model config verifies a Qwen2.5-VL architecture in float16 with a 2048 runtime context override, but the reviewed files do not provide a complete reproducible Qwen2.5-VL LoRA SFT command.

3. Later SID-Set 2K ThinkFake-style data-generation extension.
   The reproducible source-code pipeline prepares balanced real/fake image data, runs two annotation stages, validates structured JSON outputs, and builds downstream multimodal SFT JSONL.

## End-to-End Workflow

The full historical project direction is:

```text
LPCVC sample solution
  -> SID-Set schema inspection and balanced sampling
  -> image-grounded ThinkFake-style annotation
  -> train_sft.jsonl
  -> Qwen-VL SFT
  -> checkpoint selection or adapter merge
  -> merged floating-point model
  -> AIMET quantization or quantization simulation
  -> ONNX export
  -> QNN artifacts
  -> Qualcomm validation or device smoke test
  -> final competition package
```

The checked files do not prove that every item above was executed in a single continuous experiment.

## Repository Structure

```text
.
|-- README.md
|-- scripts/
|   |-- download_sidset_2k.py
|   |-- prepare_sidset_2k.py
|   |-- generate_stage1.py
|   |-- generate_stage2.py
|   |-- validate_json_outputs.py
|   |-- build_train_jsonl.py
|   `-- common.py
|-- configs/
|   `-- prompts/
|       |-- stage1_prompts.json
|       `-- stage2_prompt.txt
|-- docker_ms_swift/
|-- training_logs/
|   |-- sft_training.log
|   `-- gkd_training.log
|-- 26LPCVC_Track3_Sample_Solution/
|-- merged_qwen2_5_vl_3b_sft_ckpt100_extracted/
|-- quantized_merged_qwen2_5_vl_3b_sft_ckpt100/
|-- example1b_quant_diff_check/
`-- context2048_rollback/
```

Only the lightweight source, configuration, logs, and reports are appropriate for public documentation. The deployment and model directories may also contain large generated artifacts that are not part of a normal public checkout.

## Competition-Era Deployment Pipeline

The sample solution under `26LPCVC_Track3_Sample_Solution/` identifies the competition task as LPCVC Track 3 AI Generated Images Detection and states that its approach is based on `Qwen/Qwen2-VL-2B-Instruct`.

Verified sample-solution evidence includes:

- AIMET-based PyTorch model optimization/export as the first deployment stage.
- ONNX-to-Qualcomm NN preparation as the second deployment stage.
- A final package layout containing a context binary, `serialized_binaries/veg.serialized.bin`, embedding weights, tokenizer, positional raw inputs, mask raw inputs, and `inputs.json`.
- `contestant_uploads/inputs.json` using `Qwen/Qwen2-VL-2B-Instruct`, a 342 by 512 vision preprocessing size, 2048 context size, and 1536 embedding size.
- `inference_multi.py` using `qai_hub` for Qualcomm AI Hub inference jobs.
- `compute_score_multi_aihub.py` comparing local floating-point outputs against device or AI Hub outputs.
- `inference_script.py` validating the upload layout and supporting both Qwen2-VL and Qwen2.5-VL style mask files.

The reviewed local files do not verify the older historical checkpoint number, training-step count, training loss, token accuracy, checkpoint size, leaderboard throughput, leaderboard score, or official placement. Those values are intentionally omitted.

## Later Qwen2.5-VL-3B SFT Iteration

The workspace includes later artifact directories named like a Qwen2.5-VL 3B checkpoint-100 merge and quantization flow. Directory names alone are not treated as proof of training details.

Verified local evidence:

- The merged model config in `merged_qwen2_5_vl_3b_sft_ckpt100_extracted/` has `architectures: ["Qwen2_5_VLForConditionalGeneration"]`, `model_type: "qwen2_5_vl"`, and `dtype: "float16"`.
- The tokenizer config contains a `context_length` value of 2048 while preserving a much larger native tokenizer maximum.
- The processor config uses `Qwen2_5_VLProcessor`.
- `context2048_rollback/reports/` documents a 2048-context deployment rollback workspace with regenerated Example1A, Example1B, Example2A, and Example2B outputs, a final package layout, and a smoke test that passed file-name validation but stopped because no adb device was attached.

The reviewed files do not verify the previously described Qwen2.5-VL LoRA SFT hyperparameter set as a reproducible local command. In particular, the checked SFT log is a Qwen2-VL-2B full SFT run, not a Qwen2.5-VL LoRA run.

## SID-Set 2K ThinkFake-Style Data-Generation Extension

The SID-Set annotation sub-pipeline prepares multimodal supervised fine-tuning data. The broader LPCVC workspace also retains historical Qwen-VL fine-tuning, checkpoint-merging, AIMET quantization, ONNX/QNN conversion, and Qualcomm deployment experiments. These experiments belong to multiple project iterations and should not be interpreted as one fully unified run unless explicitly supported by local evidence.

The current reproducible data-generation extension performs:

1. Official SID-Set schema inspection.
2. Detection of the image and label columns.
3. Real, fully synthetic, and tampered label handling.
4. Balanced sampling of 1000 real and 1000 fully synthetic fake images.
5. Tampered-image exclusion by default.
6. RGB JPEG conversion with quality 95.
7. Manifest generation.
8. Stage 1 visual evidence extraction.
9. Stage 2 structured reasoning synthesis.
10. JSON validation.
11. `train_sft.jsonl` generation for downstream multimodal SFT.

Local metadata confirms the selected 2K extension contains 1000 real and 1000 fully synthetic fake images with `include_tampered: false`. This later 2K pipeline should not be described as the exact original competition training dataset unless separate competition-era evidence is provided.

Historical project notes may describe an earlier, smaller competition-era SID-Set subset. The checked public files verify the later 2K extension, so this README keeps the competition-era dataset history separate from the currently reproducible annotation pipeline.

## Stage 1 and Stage 2 Annotation

Stage 1 and Stage 2 are dependent stages.

`scripts/generate_stage1.py` reads `manifest.jsonl`, calls a vision-capable Responses API model, and writes preliminary visual evidence into `stage1_raw.jsonl`, `stage1_valid.jsonl`, and `stage1_failed.jsonl`. The Stage 1 prompt set covers eight forensic criteria across three prompts:

- edges, boundaries, texture, resolution, material, and object detail;
- physical logic, text/symbol quality, and human/biological structure;
- lighting, shadows, perspective, and spatial relationships.

`scripts/generate_stage2.py` reads `stage1_valid.jsonl`. The Stage 2 prompt inserts the three Stage 1 outputs into the final synthesis prompt, then calls a text Responses API model. The resulting pipeline is:

```text
image
  -> preliminary visual evidence extraction
  -> Stage 1 result
  -> refined 8-criterion structured reasoning
  -> final real/fake judgment
```

Both stages support `--resume`.

## train_sft.jsonl Construction

`scripts/build_train_jsonl.py` consumes:

- `manifest.jsonl`
- `stage1_valid.jsonl`
- `stage2_valid.jsonl`

Each output record has this schema:

```json
{
  "id": "sample id",
  "image": "path to image",
  "label": "REAL or AI-GENERATED",
  "messages": [
    {
      "role": "user",
      "content": "<image>\nIs this image real or generated by AI? Please analyze visual forensic evidence and output the structured JSON judgment."
    },
    {
      "role": "assistant",
      "content": "{\"per_criterion\":[...],\"overall_likelihood\":\"Real or AI-Generated\"}"
    }
  ],
  "thinkfake_stage1": {
    "prompt1": "Stage 1 text",
    "prompt2": "Stage 1 text",
    "prompt3": "Stage 1 text"
  },
  "generator_model": "annotation model name",
  "reasoning_effort": "reasoning effort label"
}
```

`train_sft.jsonl` is not merely a binary label file. It contains an image reference, a multimodal user instruction, a structured assistant response, preserved Stage 1 evidence, and a final real/fake judgment inside the Stage 2 JSON.

GPT-generated rationales are synthetic supervision. They may contain hallucinated or weakly grounded forensic observations, so schema validation and manual sampling are required before training.

## Supervised Fine-Tuning Role

The reviewed `training_logs/sft_training.log` verifies one MS-Swift SFT run using `Qwen/Qwen2-VL-2B-Instruct`, not Qwen2.5-VL. Verified settings from that log include:

- `swift/cli/sft.py`
- `--model_type qwen2_vl`
- `--model Qwen/Qwen2-VL-2B-Instruct`
- `--tuner_type full`
- `--freeze_vit true`
- `--freeze_aligner false`
- `--bf16 true`
- `--num_train_epochs 3.0`
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 16`
- `--eval_steps 100`
- `--save_steps 100`
- `--save_total_limit 2`
- `--max_length 2048`
- `--max_new_tokens 2048`
- `--learning_rate 2e-5`

This log also references SID-style train/validation JSONL splits, but it does not by itself establish final competition submission selection.

## Checkpoint Merge and Deployment Artifacts

Use these terms precisely:

- SFT adapter or checkpoint: the training output selected for later use.
- Merged floating-point model: a model directory after adapter/checkpoint merging, before deployment quantization.
- AIMET quantization simulation or quantized model: the model state used for calibration, simulation, or quantized export.
- ONNX export: the intermediate graph export consumed by downstream Qualcomm tooling.
- QNN artifacts: Qualcomm runtime binaries, serialized context files, raw masks, raw position IDs, and related runtime assets.
- Qualcomm validation result: AI Hub, device, or sample-script validation output.
- Final competition package: the uploadable package assembled from generated deployment artifacts.

`example1b_quant_diff_check/` is documented only as a floating-point versus quantized-output comparison workspace. Its report states that local `llm_inout.py` generation produced `inputs*.pt` and `outputs*.pt`, but AI Hub inference and `compute_score_multi_aihub.py` were not run because a real AI Hub model id and `qai_hub` installation were missing.

`context2048_rollback/` is documented as a deployment-compatibility rollback workspace. The local reports support a 2048-context regenerated package and file-name smoke-test validation, but they also record a residual verifier warning and a device-test stop due to no attached adb device.

## Experimental GKD Work

`training_logs/gkd_training.log` contains an MS-Swift RLHF/GKD command using `Qwen/Qwen2-VL-2B-Instruct`, a teacher checkpoint path, full tuning, frozen vision tower, unfrozen aligner, bf16, 2048 max length, and 3 epochs.

No reviewed artifact proves that this GKD run was used in the selected LPCVC submission. It is therefore documented as experimental work rather than part of the final pipeline.

## Installation

Use a clean Python environment for the annotation scripts:

```bash
PROJECT_ROOT="/path/to/LPCVC-Track3"
cd "$PROJECT_ROOT"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install datasets huggingface_hub pillow tqdm openai
```

The MS-Swift Docker files are retained for training-environment reference. Qualcomm deployment requires separate proprietary Qualcomm and AIMET/QNN tooling that is not vendored in this public repository.

## Hugging Face Authentication

If the SID-Set dataset requires authentication in your environment, authenticate before scanning or downloading:

```bash
huggingface-cli login
```

or set a token in the environment:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Dataset Schema Scan

Inspect the upstream SID-Set schema without downloading images:

```bash
PROJECT_ROOT="/path/to/LPCVC-Track3"
SIDSET_ROOT="$PROJECT_ROOT/dataset/SID_Set_2k"
cd "$PROJECT_ROOT"

python scripts/download_sidset_2k.py \
  --hf_dataset saberzl/SID_Set \
  --output_root "$SIDSET_ROOT" \
  --scan_only
```

The local metadata shows the detected image column as `image`, label column as `label`, and label categories for real, full synthetic, and tampered images.

## Balanced SID-Set Download

Download the balanced 2K subset:

```bash
python scripts/download_sidset_2k.py \
  --hf_dataset saberzl/SID_Set \
  --output_root "$SIDSET_ROOT" \
  --num_real 1000 \
  --num_fake 1000 \
  --seed 42 \
  --resume
```

By default, tampered images are excluded. Use `--include_tampered true` only if you intentionally want tampered samples included in the fake class.

## Manifest Preparation

Build the annotation workspace manifest and empty output files:

```bash
python scripts/prepare_sidset_2k.py \
  --sidset_root "$SIDSET_ROOT" \
  --output_dir output_sidset_thinkfake_gpt55_xhigh \
  --num_real 1000 \
  --num_fake 1000 \
  --seed 42 \
  --resume
```

The manifest rows contain sample ids, image paths, labels, label ids, source dataset metadata, and original SID-Set label metadata when available.

## Four-Image Smoke Test

Before running a paid annotation pass, create a four-image workspace:

```bash
python scripts/prepare_sidset_2k.py \
  --sidset_root "$SIDSET_ROOT" \
  --output_dir output_sidset_thinkfake_gpt55_xhigh_smoke \
  --num_real 2 \
  --num_fake 2 \
  --seed 42 \
  --resume
```

Run Stage 1, Stage 2, validation, and JSONL construction on this smoke workspace first.

## Stage 1 Annotation Command

Stage 1 calls an external Responses API model with image input. This can incur API cost.

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-5.5"

python scripts/generate_stage1.py \
  --output_dir output_sidset_thinkfake_gpt55_xhigh_smoke \
  --model "$OPENAI_MODEL" \
  --api_key_env OPENAI_API_KEY \
  --reasoning_effort xhigh \
  --max_output_tokens_stage1 500 \
  --resume
```

The local script enforces `--max_output_tokens_stage1 500`.

## Stage 2 Annotation Command

Stage 2 reads `stage1_valid.jsonl` and synthesizes the final structured JSON:

```bash
python scripts/generate_stage2.py \
  --output_dir output_sidset_thinkfake_gpt55_xhigh_smoke \
  --model "$OPENAI_MODEL" \
  --api_key_env OPENAI_API_KEY \
  --reasoning_effort xhigh \
  --max_output_tokens_stage2 1200 \
  --resume
```

The Stage 2 prompt requires exactly eight criteria and a parseable JSON object.

## Validation

Validate Stage 2 outputs:

```bash
python scripts/validate_json_outputs.py \
  --output_dir output_sidset_thinkfake_gpt55_xhigh_smoke
```

Validation checks the eight-criterion schema, key order, criterion names, `aigc score` values, final likelihood, and consistency with the manifest label.

## SFT JSONL Generation

Build downstream multimodal SFT data:

```bash
python scripts/build_train_jsonl.py \
  --output_dir output_sidset_thinkfake_gpt55_xhigh_smoke
```

The resulting `train_sft.jsonl` can be consumed by a multimodal SFT workflow after manual review and any framework-specific conversion.

## Output Structure

The annotation output directory is expected to contain:

```text
output_sidset_thinkfake_gpt55_xhigh/
|-- manifest.jsonl
|-- stage1_raw.jsonl
|-- stage1_valid.jsonl
|-- stage1_failed.jsonl
|-- stage2_raw.jsonl
|-- stage2_valid.jsonl
|-- stage2_invalid.jsonl
|-- train_sft.jsonl
`-- logs/
```

Regenerate or normalize `manifest.jsonl` if it contains absolute paths from another workstation.

## Security and API-Cost Notes

- Do not commit API keys, Hugging Face tokens, generated manifests with private paths, or paid API outputs that cannot be published.
- Stage 1 and Stage 2 call external APIs and can be expensive at 2K-image scale.
- The OpenAI client code uses `store=False`, but you are still responsible for reviewing provider terms and data-handling requirements.
- Run the four-image smoke test before launching a full annotation job.
- Keep model checkpoints, datasets, QNN binaries, generated raw files, and submission zips out of the public repository unless their licenses and competition rules allow release.

## Limitations

- The public files reviewed here do not prove an official competition placement.
- The reviewed local files do not verify the older historical checkpoint and leaderboard metrics.
- The checked SFT log verifies Qwen2-VL-2B full SFT, not the previously described Qwen2.5-VL LoRA SFT settings.
- Synthetic rationales can hallucinate visual evidence, even when prompts ask for grounded observations.
- The deployment reports include residual validation warnings and device-test limitations.
- The repository does not include proprietary Qualcomm tooling or generated binary artifacts required to reproduce a full device package from scratch.

## Third-Party Licenses

This project depends on third-party datasets, models, and tooling. Check and comply with the upstream licenses and terms for:

- LPCVC sample solution materials and competition rules.
- SID-Set data.
- Qwen/Qwen-VL model weights and tokenizer assets.
- MS-Swift and training dependencies.
- AIMET, QNN SDK, Qualcomm AI Hub, and related Qualcomm deployment tooling.
- OpenAI API usage for synthetic annotation.

## Acknowledgements

This work builds on the LPCVC Track 3 sample solution, SID-Set, ThinkFake-style structured forensic annotation ideas, Qwen-VL models, MS-Swift training tooling, AIMET/QNN deployment tooling, Qualcomm AI Hub utilities, and OpenAI Responses API annotation support.
