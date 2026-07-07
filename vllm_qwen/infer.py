import argparse
import io
import json
import os
import tarfile
import time
from itertools import islice
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

PROMPT = """
You are an expert image annotation model for semantic segmentation.

Carefully inspect the input image and identify only the visual categories that are truly visible and suitable for image segmentation. Do not describe the image. Do not generate captions. Output only category labels.

For each visible category, output three levels of labels:

1. parent_label:
   A high-level semantic parent category.
   It should describe the general group that the category belongs to.

2. broad_label:
   A stable and general semantic segmentation label.
   It should be suitable for training a segmentation model and should be close to the category granularity used in common semantic segmentation datasets such as ADE20K / ADE150 and COCO-Stuff.

3. specific_label:
   A more specific category name when the visual evidence is clear.
   The specific_label should be more detailed than broad_label, but it must still be a concise and common visual category name.
   If the specific category is uncertain, use the same value as broad_label.

Definitions:

Foreground labels, or thing classes, are countable objects with relatively independent instance boundaries.
Examples:
person, car, bicycle, motorcycle, bus, truck, dog, cat, chair, table, sofa, bed, bottle, bag, boat.

Background labels, or stuff classes, are continuous regions, scene surfaces, materials, or large uncountable areas.
Examples:
sky, road, sidewalk, grass, wall, floor, ceiling, water, sand, building, mountain, tree, sea, river.

Label granularity rules:

* Do not use overly broad labels when a stable segmentation category is visible.
  For example, use "car" instead of "vehicle", "chair" instead of "furniture", and "tree" instead of "plant".

* Do not use arbitrary fine-grained names.
  Use a specific_label only when the image provides clear visual evidence and the label is a common visual category.

* The broad_label should be stable and reusable for segmentation training.
  Examples:
  "swivel chair", "armchair", "office chair" -> broad_label: "chair"
  "coffee table", "desk", "dining table" -> broad_label: "table"
  "wardrobe", "cupboard", "storage cabinet" -> broad_label: "cabinet"
  "skyscraper", "house", "apartment building" -> broad_label: "building"
  "tile floor", "wooden floor", "marble floor" -> broad_label: "floor"
  "asphalt road", "street", "lane" -> broad_label: "road"

* The specific_label may preserve useful fine-grained information.
  Good examples:
  broad_label: "chair", specific_label: "swivel chair"
  broad_label: "chair", specific_label: "armchair"
  broad_label: "table", specific_label: "desk"
  broad_label: "table", specific_label: "coffee table"
  broad_label: "cabinet", specific_label: "wardrobe"
  broad_label: "shelf", specific_label: "bookcase"
  broad_label: "building", specific_label: "skyscraper"
  broad_label: "floor", specific_label: "tile floor"
  broad_label: "tree", specific_label: "palm"
  broad_label: "person", specific_label: "person"

Parent label examples:

* person -> person
* dog, cat, horse, bird, cow, sheep -> animal
* car, bus, truck, bicycle, motorcycle, boat, airplane, train -> vehicle
* chair, armchair, swivel chair, table, desk, sofa, bed, cabinet, shelf, wardrobe -> furniture
* refrigerator, stove, oven, washer, microwave -> appliance
* television, monitor, computer, laptop, phone -> electronic
* bottle, cup, bowl, plate -> tableware
* bag, basket, box, barrel -> container
* shirt, pants, dress, hat, shoe -> clothing
* tree, palm, flower, grass, plant -> plant
* building, house, skyscraper, tower -> building
* door, window, stairs, railing, column, fence -> structural element
* road, sidewalk, floor, wall, ceiling, countertop -> surface
* sky -> sky
* water, sea, river, lake -> water region
* mountain, rock, sand, earth, field -> natural region
* signboard, traffic light, poster -> sign
* other visible countable objects -> object

Important constraints:

1. Output only categories that are truly visible in the image.
2. Do not infer invisible objects from context or common sense.
3. Do not output tiny, blurry, heavily occluded, or uncertain objects.
4. Do not output scene descriptions such as "street scene", "kitchen", "indoor", "outdoor", or "park".
5. Do not output actions, relations, colors, sizes, styles, brands, or subjective attributes.
   Bad examples:
   "red chair", "wooden table", "large building", "old wall", "sitting person", "busy road".
6. Do not output duplicate synonyms for the same semantic object.
   Bad examples:
   "person", "man", "human" for the same category.
   "car", "vehicle", "sedan" for the same category.
   "tree", "plant", "palm tree" for the same category.
7. For humans and animals, prefer the whole-body category.
   Do not output body parts such as "hand", "face", "arm", "leg", "tail", or "wheel" unless the part is very prominent, clearly segmented, and semantically important.
8. Use concise English singular nouns or short noun phrases.
   Use "person" instead of "people".
   Use "tree" instead of "trees".
   Use "car" instead of "cars".
9. If no suitable label exists for either foreground or background, output an empty list [].
10. The output must be valid JSON only. Do not output explanations, comments, markdown, or extra text.

Output format:

{
"foreground_labels": [
{
"parent_label": "furniture",
"broad_label": "chair",
"specific_label": "swivel chair"
}
],
"background_labels": [
{
"parent_label": "surface",
"broad_label": "floor",
"specific_label": "tile floor"
}
]
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
    p = argparse.ArgumentParser(description="Process one or more SA-1B tar shards with vLLM.")
    p.add_argument("--tar-dir", default="/data_16T/tc/tiancong/OpenDataLab___SA-1B/sa1b")
    p.add_argument("--tar-glob", default="sa_*.tar")
    p.add_argument(
        "--tar-files",
        nargs="+",
        default=None,
        help="指定要顺序处理的 tar 文件名列表，例如: sa_000014.tar sa_000016.tar sa_000017.tar sa_000019.tar",
    )
    p.add_argument("--model-path", default="/data_16T/tc/tiancong/OpenDataLab___SA-1B/qwen3.6/Qwen3.6-27B")
    p.add_argument("--output-dir", default="/data_16T/tc/tiancong/OpenDataLab___SA-1B/qwen3.6/output/sa1b_tar_vllm")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-image-size", type=int, default=1280)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--tar-limit", type=int, default=0)
    p.add_argument("--limit-per-tar", type=int, default=0)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=2512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--worker-name", default="worker")
    return p.parse_args()


def resize_image(image: Image.Image, max_image_size: int):
    image = image.convert("RGB")
    original_size = image.size
    w, h = original_size
    if max_image_size <= 0 or max(w, h) <= max_image_size:
        return image.copy(), original_size, original_size, False
    scale = max_image_size / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), original_size, new_size, True


def image_members(tar: tarfile.TarFile):
    return [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(IMAGE_EXTS)]


def read_done(path: Path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_key = item.get("image")
            if image_key:
                done.add(image_key)
    return done


def shard_tar_paths(tar_paths, num_shards: int, shard_index: int):
    if num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return tar_paths[shard_index::num_shards]


def load_processor_and_llm(args):
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )
    return processor, llm


def build_prompt(processor, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def process_tar(tar_path: Path, args, processor, llm):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{tar_path.stem}.jsonl"
    done = read_done(out_jsonl)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    with tarfile.open(tar_path, "r") as tar, out_jsonl.open("a", encoding="utf-8") as fout:
        members = [m for m in image_members(tar) if f"{tar_path.stem}/{m.name}" not in done]
        if args.limit_per_tar > 0:
            members = members[:args.limit_per_tar]

        total_images = len(members)
        print(f"{tar_path.name}: done={len(done)} remaining={total_images}")
        progress = tqdm(
            total=total_images,
            desc=tar_path.stem,
            unit="img",
            dynamic_ncols=True,
        )
        tar_start = time.time()
        processed_images = 0

        for batch_idx, batch_members in enumerate(batched(members, args.batch_size), start=1):
            batch_prompts = []
            batch_meta = []

            for member in batch_members:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with Image.open(io.BytesIO(extracted.read())) as image:
                    input_image, original_size, input_size, resized = resize_image(image, args.max_image_size)
                prompt = build_prompt(processor, input_image)
                batch_prompts.append({"prompt": prompt, "multi_modal_data": {"image": input_image}})
                batch_meta.append(
                    {
                        "image": f"{tar_path.stem}/{member.name}",
                        "tar": str(tar_path),
                        "member": member.name,
                        "original_size": list(original_size),
                        "input_size": list(input_size),
                        "resized": resized,
                    }
                )

            if not batch_prompts:
                continue

            start = time.time()
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            batch_time = time.time() - start

            for meta, output in zip(batch_meta, outputs):
                result = meta | {"output": output.outputs[0].text.strip()}
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            batch_count = len(batch_meta)
            processed_images += batch_count
            elapsed = time.time() - tar_start
            imgs_per_sec = processed_images / elapsed if elapsed > 0 else 0.0
            avg_batch_time = elapsed / batch_idx if batch_idx > 0 else 0.0

            progress.update(batch_count)
            progress.set_postfix(
                batch=batch_idx,
                batch_time=f"{batch_time:.2f}s",
                avg_batch=f"{avg_batch_time:.2f}s",
                imgs=f"{processed_images}/{total_images}",
                imgs_per_s=f"{imgs_per_sec:.2f}",
            )

        progress.close()


def main():
    args = parse_args()
    if args.tar_files:
        tar_paths = [Path(args.tar_dir) / tar_name for tar_name in args.tar_files]
    else:
        tar_paths = sorted(Path(args.tar_dir).glob(args.tar_glob))
        if args.tar_limit > 0:
            tar_paths = tar_paths[:args.tar_limit]

    if not tar_paths:
        raise FileNotFoundError(f"No tar files found in {args.tar_dir} matching {args.tar_glob}")

    missing = [str(path) for path in tar_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"These tar files do not exist: {missing}")

    shard_paths = shard_tar_paths(tar_paths, args.num_shards, args.shard_index)
    worker_name = args.worker_name or f"worker{args.shard_index}"

    print(f"tar_dir={args.tar_dir}")
    print(f"num_tars={len(tar_paths)}")
    print(f"num_shards={args.num_shards}")
    print(f"shard_index={args.shard_index}")
    print(f"assigned_tars={len(shard_paths)}")
    print(f"output_dir={args.output_dir}")
    print(f"worker_name={worker_name}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    if not shard_paths:
        print(f"[{worker_name}] no tar files assigned, exiting.")
        return

    processor, llm = load_processor_and_llm(args)

    for tar_path in shard_paths:
        process_tar(tar_path, args, processor, llm)


if __name__ == "__main__":
    main()
