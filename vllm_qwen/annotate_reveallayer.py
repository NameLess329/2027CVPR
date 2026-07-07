import argparse
import json
import os
import re
import tempfile
import time
from itertools import islice
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


ANNOTATION_PROMPT = """
You are an expert image prompt annotator for layered image decomposition datasets.

You will receive multiple images from one sample in this exact order:
1. foreground RGBA layers: layer_0, layer_1, ... from top to bottom
2. background image
3. full composited image

Task:
Write concise English generation prompts for every provided image.

Requirements:
- For each foreground RGBA layer, describe only the non-transparent visible foreground object or region in that layer.
- Foreground layer prompts must not include any background, scene, environment, floor, wall, sky, road, landscape, context, or objects that are only visible outside the foreground alpha region.
- If a foreground layer contains only one separated object, describe that object directly instead of describing it as part of a scene.
- For the background image, describe the clean background image without the explicitly separated foreground layer objects.
- Background prompts must exclude only the known separated foreground layers: layer_0, layer_1, ...
- If the background image still contains other visible objects or regions that were not separated into layers, they may be described normally.
- For the full image, describe the complete composited image and explicitly include the background plus every separated foreground layer in their visible spatial/semantic relationship.
- Use natural, reusable image-generation prompt language.
- Mention important object categories, scene context, layout, and visible appearance when useful.
- Do not invent objects that are not visible.
- Do not mention file names, layer indices, alpha channels, masks, bounding boxes, or dataset metadata.
- Do not describe transparency, cutout edges, empty canvas, checkerboard patterns, or missing pixels.
- Output valid JSON only. Do not output markdown, comments, or explanations.

Return this JSON schema exactly:
{
  "layers": [
    {"prompt": "prompt for layer_0"},
    {"prompt": "prompt for layer_1"}
  ],
  "background": {"prompt": "prompt for the clean background"},
  "global": {"prompt": "prompt for the full composited image"}
}
"""

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate RevealLayer layered images with Qwen/vLLM prompts."
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        default=r"E:\Image_Decomposition\code\data\RevealLayer-100K\Benchmark\RevealLayerBenchMark-200",
        help="RevealLayer dataset root containing sample folders and metaData.json.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata JSON. Defaults to <dataset-root>/metaData.json.",
    )
    parser.add_argument(
        "--output-json",
        "--output_json",
        default=None,
        help="Output flattened annotation JSON. Defaults to <dataset-root>/prompt_annotations.json.",
    )
    parser.add_argument(
        "--raw-output-jsonl",
        "--raw_output_jsonl",
        default=None,
        help="Optional sidecar JSONL for raw model responses. Defaults to <output-json>.raw.jsonl.",
    )
    parser.add_argument("--model-path", "--model_path", required=True, help="Local Qwen model path.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--max-image-size", "--max_image_size", type=int, default=1280)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max-model-len", "--max_model_len", type=int, default=8192)
    parser.add_argument(
        "--max-num-seqs",
        "--max_num_seqs",
        type=int,
        default=32,
        help="Maximum concurrent decode sequences for vLLM. Keep this low for offline batch annotation.",
    )
    parser.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N metadata samples.")
    parser.add_argument("--resume", action="store_true", help="Skip imgids already present in output JSON.")
    parser.add_argument("--num-shards", "--num_shards", type=int, default=1)
    parser.add_argument("--shard-index", "--shard_index", type=int, default=0)
    parser.add_argument(
        "--save-every",
        "--save_every",
        type=int,
        default=1,
        help="Atomically rewrite output JSON every N processed samples.",
    )
    return parser.parse_args()


def load_metadata(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Metadata must be a JSON list: {path}")
    return data


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        temp_name = f.name
    os.replace(temp_name, path)


def read_existing_records(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Existing output must be a JSON list: {path}")
    return data


def completed_imgids(records):
    return {record["imgid"] for record in records if record.get("item_type") == "global" and record.get("imgid")}


def shard_items(items, num_shards, shard_index):
    if num_shards <= 0:
        raise ValueError("--num-shards must be greater than 0")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return items[shard_index::num_shards]


def resize_image(image, max_image_size):
    image = image.convert("RGB")
    original_size = image.size
    width, height = original_size
    if max_image_size <= 0 or max(width, height) <= max_image_size:
        return image.copy(), original_size, original_size, False
    scale = max_image_size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), original_size, new_size, True


def load_image(dataset_root, relative_path, max_image_size):
    path = Path(dataset_root) / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image extension: {path}")
    with Image.open(path) as image:
        return resize_image(image, max_image_size)


def get_layer_paths(item):
    layers = item.get("LayerInfoRaw") or item.get("layers") or []
    if not isinstance(layers, list):
        raise ValueError(f"LayerInfoRaw must be a list for imgid={item.get('imgid')}")
    return layers


def get_bboxes(item, num_layers):
    detections = item.get("detections") or []
    bboxes = []
    for index in range(num_layers):
        bbox = None
        if index < len(detections) and isinstance(detections[index], dict):
            bbox = detections[index].get("bbox")
        bboxes.append(bbox)
    return bboxes


def metadata_from_item(item):
    imgid = str(item.get("imgid", ""))
    layer_paths = get_layer_paths(item)
    background_path = item.get("background")
    full_image_path = item.get("full_image")
    if not imgid or not background_path or not full_image_path:
        raise ValueError(f"Missing imgid/background/full_image in metadata item: {item}")
    return {
        "imgid": imgid,
        "layer_paths": layer_paths,
        "background_path": background_path,
        "full_image_path": full_image_path,
        "bboxes": get_bboxes(item, len(layer_paths)),
    }


def build_request(processor, dataset_root, item, max_image_size):
    meta = metadata_from_item(item)

    image_specs = []
    for index, path in enumerate(meta["layer_paths"]):
        image_specs.append({"item_type": "layer", "layer_index": index, "image": path})
    image_specs.append({"item_type": "background", "layer_index": None, "image": meta["background_path"]})
    image_specs.append({"item_type": "global", "layer_index": None, "image": meta["full_image_path"]})

    images = []
    image_meta = []
    for spec in image_specs:
        image, original_size, input_size, resized = load_image(dataset_root, spec["image"], max_image_size)
        images.append(image)
        image_meta.append(
            spec
            | {
                "original_size": list(original_size),
                "input_size": list(input_size),
                "resized": resized,
            }
        )

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": ANNOTATION_PROMPT})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return {
        "prompt": prompt,
        "multi_modal_data": {"image": images},
        "metadata": meta | {"image_meta": image_meta},
    }


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_prompt(value):
    if isinstance(value, dict):
        value = value.get("prompt", "")
    if value is None:
        return ""
    return str(value).strip()


def records_from_response(meta, response_text):
    parsed = extract_json_object(response_text)
    imgid = meta["imgid"]
    bboxes = meta["bboxes"]
    layer_paths = meta["layer_paths"]
    layers = parsed.get("layers", [])
    if not isinstance(layers, list):
        layers = []

    records = []
    for index, path in enumerate(layer_paths):
        layer_prompt = normalize_prompt(layers[index] if index < len(layers) else "")
        records.append(
            {
                "imgid": imgid,
                "item_type": "layer",
                "layer_index": index,
                "image": path,
                "prompt": layer_prompt,
                "bbox": bboxes[index] if index < len(bboxes) else None,
            }
        )

    records.append(
        {
            "imgid": imgid,
            "item_type": "background",
            "layer_index": None,
            "image": meta["background_path"],
            "prompt": normalize_prompt(parsed.get("background")),
            "bbox": None,
        }
    )
    records.append(
        {
            "imgid": imgid,
            "item_type": "global",
            "layer_index": None,
            "image": meta["full_image_path"],
            "prompt": normalize_prompt(parsed.get("global")),
            "bbox": bboxes,
        }
    )
    return records


def error_records(meta, response_text, error):
    records = []
    imgid = meta["imgid"]
    for index, path in enumerate(meta["layer_paths"]):
        records.append(
            {
                "imgid": imgid,
                "item_type": "layer",
                "layer_index": index,
                "image": path,
                "prompt": "",
                "bbox": meta["bboxes"][index] if index < len(meta["bboxes"]) else None,
                "error": str(error),
                "raw_response": response_text,
            }
        )
    records.append(
        {
            "imgid": imgid,
            "item_type": "background",
            "layer_index": None,
            "image": meta["background_path"],
            "prompt": "",
            "bbox": None,
            "error": str(error),
            "raw_response": response_text,
        }
    )
    records.append(
        {
            "imgid": imgid,
            "item_type": "global",
            "layer_index": None,
            "image": meta["full_image_path"],
            "prompt": "",
            "bbox": meta["bboxes"],
            "error": str(error),
            "raw_response": response_text,
        }
    )
    return records


def load_processor_and_llm(args):
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )
    return processor, llm


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    metadata_path = Path(args.metadata) if args.metadata else dataset_root / "metaData.json"
    output_json = Path(args.output_json) if args.output_json else dataset_root / "prompt_annotations.json"
    raw_output_jsonl = (
        Path(args.raw_output_jsonl)
        if args.raw_output_jsonl
        else output_json.with_suffix(output_json.suffix + ".raw.jsonl")
    )

    records = read_existing_records(output_json) if args.resume else []
    done = completed_imgids(records) if args.resume else set()

    metadata = load_metadata(metadata_path)
    if args.limit > 0:
        metadata = metadata[: args.limit]
    metadata = shard_items(metadata, args.num_shards, args.shard_index)
    metadata = [item for item in metadata if str(item.get("imgid", "")) not in done]

    print(f"dataset_root={dataset_root}")
    print(f"metadata={metadata_path}")
    print(f"output_json={output_json}")
    print(f"raw_output_jsonl={raw_output_jsonl}")
    print(f"samples_to_process={len(metadata)}")
    print(f"resume_done={len(done)}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    if not metadata:
        atomic_write_json(output_json, records)
        return

    processor, llm = load_processor_and_llm(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    processed_since_save = 0
    raw_output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with raw_output_jsonl.open("a", encoding="utf-8") as raw_f:
        progress = tqdm(total=len(metadata), desc="annotating", unit="sample", dynamic_ncols=True)
        start_time = time.time()

        for batch in batched(metadata, args.batch_size):
            requests = []
            for item in batch:
                try:
                    request = build_request(processor, dataset_root, item, args.max_image_size)
                    requests.append(request)
                except Exception as exc:
                    imgid = str(item.get("imgid", ""))
                    raw_f.write(json.dumps({"imgid": imgid, "error": str(exc)}, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    try:
                        records.extend(error_records(metadata_from_item(item), "", exc))
                        processed_since_save += 1
                        if processed_since_save >= args.save_every:
                            atomic_write_json(output_json, records)
                            processed_since_save = 0
                    except Exception:
                        pass
                    progress.update(1)

            if not requests:
                continue

            outputs = llm.generate(
                [{"prompt": req["prompt"], "multi_modal_data": req["multi_modal_data"]} for req in requests],
                sampling_params,
                use_tqdm=False,
            )

            for request, output in zip(requests, outputs):
                meta = request["metadata"]
                response_text = output.outputs[0].text.strip()
                raw_f.write(
                    json.dumps(
                        {"imgid": meta["imgid"], "raw_response": response_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_f.flush()

                try:
                    records.extend(records_from_response(meta, response_text))
                except Exception as exc:
                    records.extend(error_records(meta, response_text, exc))

                processed_since_save += 1
                progress.update(1)

                elapsed = max(time.time() - start_time, 1e-6)
                progress.set_postfix(samples_per_s=f"{progress.n / elapsed:.3f}")

                if processed_since_save >= args.save_every:
                    atomic_write_json(output_json, records)
                    processed_since_save = 0

        progress.close()

    atomic_write_json(output_json, records)


if __name__ == "__main__":
    main()
