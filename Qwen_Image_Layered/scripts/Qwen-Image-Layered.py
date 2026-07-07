import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen-Image-Layered with local weights."
    )
    parser.add_argument(
        "--weight-dir",
        required=True,
        help="Local base directory of weights, for example: /home/user/lhr/wksps_image/weight",
    )
    parser.add_argument(
        "--input-image",
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt for image decomposition",
    )
    parser.add_argument(
        "--layer-num",
        type=int,
        required=True,
        help="Number of layers to decompose",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Output image height, must be divisible by 16",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Output image width, must be divisible by 16",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed, default: 1",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Inference steps, default: 50",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, default: cuda",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("Both --height and --width must be divisible by 16")
    if args.layer_num <= 0:
        raise ValueError("--layer-num must be greater than 0")

    input_image_path = Path(args.input_image)
    if not input_image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    weight_dir = Path(args.weight_dir)
    expected_dirs = [
        weight_dir / "Qwen" / "Qwen-Image-Layered" / "transformer",
        weight_dir / "Qwen" / "Qwen-Image-Layered" / "vae",
        weight_dir / "Qwen" / "Qwen-Image" / "text_encoder",
        weight_dir / "Qwen" / "Qwen-Image" / "tokenizer",
    ]
    missing_dirs = [str(path) for path in expected_dirs if not path.exists()]
    if missing_dirs:
        missing_text = "\n".join(missing_dirs)
        raise FileNotFoundError(
            "Local weight directory is incomplete. Missing paths:\n"
            f"{missing_text}"
        )


def build_model_configs(weight_dir: Path) -> list[ModelConfig]:
    return [
        ModelConfig(
            model_id="Qwen/Qwen-Image-Layered",
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            local_model_path=str(weight_dir),
            skip_download=True,
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="text_encoder/model*.safetensors",
            local_model_path=str(weight_dir),
            skip_download=True,
        ),
        ModelConfig(
            model_id="Qwen/Qwen-Image-Layered",
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            local_model_path=str(weight_dir),
            skip_download=True,
        ),
    ]


def build_tokenizer_config(weight_dir: Path) -> ModelConfig:
    return ModelConfig(
        model_id="Qwen/Qwen-Image",
        origin_file_pattern="tokenizer/",
        local_model_path=str(weight_dir),
        skip_download=True,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    weight_dir = Path(args.weight_dir)
    input_image_path = Path(args.input_image)
    output_dir = Path(args.output_dir)

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=build_model_configs(weight_dir),
        tokenizer_config=build_tokenizer_config(weight_dir),
        # processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
    )

    input_image = Image.open(input_image_path).convert("RGBA").resize((args.width, args.height))
    images = pipe(
        args.prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        layer_input_image=input_image,
        layer_num=args.layer_num,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        if i == 0:
            continue
        image.save(output_dir / f"image_{i}.png")


if __name__ == "__main__":
    main()
