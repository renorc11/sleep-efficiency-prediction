import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models" / "biomistral-v3-adapter"
DEST = ROOT / "models" / "biomistral-v3-mlx-adapter"

source_weights = SOURCE / "adapter_model.safetensors"
source_config = SOURCE / "adapter_config.json"

if not source_weights.is_file() or not source_config.is_file():
    raise FileNotFoundError(
        "The original adapter files were not found. "
        f"Expected them under: {SOURCE}"
    )

with source_config.open("r", encoding="utf-8") as f:
    hf_config = json.load(f)

if hf_config.get("peft_type") != "LORA":
    raise ValueError("This converter supports standard LoRA adapters only.")

if hf_config.get("use_dora", False):
    raise ValueError("DoRA adapters are not supported by this converter.")

rank = int(hf_config["r"])
alpha = float(hf_config["lora_alpha"])
dropout = float(hf_config.get("lora_dropout", 0.0))
scale = alpha / rank

projection_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
converted = {}
seen = set()

with safe_open(source_weights, framework="np") as f:
    for key in f.keys():
        parts = key.split(".")

        layer_index = next(
            (
                int(parts[i + 1])
                for i, part in enumerate(parts[:-1])
                if part == "layers" and parts[i + 1].isdigit()
            ),
            None,
        )

        if layer_index is None:
            raise ValueError(f"Could not identify transformer layer: {key}")

        projection = next(
            (name for name in projection_names if name in parts),
            None,
        )

        if projection is None:
            raise ValueError(f"Unexpected projection in tensor: {key}")

        if key.endswith(".lora_A.weight"):
            kind = "lora_a"
        elif key.endswith(".lora_B.weight"):
            kind = "lora_b"
        else:
            raise ValueError(f"Unexpected tensor name: {key}")

        # MLX stores the LoRA matrices in the transposed orientation
        # compared with the standard Hugging Face PEFT tensors.
        tensor = np.ascontiguousarray(f.get_tensor(key).T)

        mlx_key = (
            f"model.layers.{layer_index}."
            f"self_attn.{projection}.{kind}"
        )

        if mlx_key in converted:
            raise ValueError(f"Duplicate converted tensor: {mlx_key}")

        converted[mlx_key] = tensor
        seen.add((layer_index, projection, kind))

expected = {
    (layer, projection, kind)
    for layer in range(32)
    for projection in projection_names
    for kind in ("lora_a", "lora_b")
}

if seen != expected:
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    raise ValueError(
        f"Adapter tensor coverage mismatch.\n"
        f"Missing: {missing[:10]}\n"
        f"Unexpected: {extra[:10]}"
    )

DEST.mkdir(parents=True, exist_ok=True)

# Use a new MLX-compatible configuration and preserve the source files.
mlx_config = {
    "fine_tune_type": "lora",
    "num_layers": 32,
    "lora_parameters": {
        "rank": rank,
        "scale": scale,
        "dropout": dropout,
        "keys": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ],
    },
}

save_file(converted, str(DEST / "adapters.safetensors"))

with (DEST / "adapter_config.json").open("w", encoding="utf-8") as f:
    json.dump(mlx_config, f, indent=2)
    f.write("\n")

print("Adapter conversion completed.")
print("Original adapter:", SOURCE)
print("Converted adapter:", DEST)
print("Tensor count:", len(converted))
print("Transformer layers: 32")
print("Projections per layer: 4")
print("Rank:", rank)
print("LoRA scale:", scale)
print("Dropout:", dropout)
print("Converted weight file size: {:.2f} MiB".format(
    (DEST / "adapters.safetensors").stat().st_size / (1024 * 1024)
))
