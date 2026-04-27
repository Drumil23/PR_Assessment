import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import json
import time
import os

print("Loading Moondream model")
model_id = "vikhyatk/moondream2"
revision = "2025-01-09"  # pin to specific revision for reproducibility

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # use float16 if on GPU
)
model.eval()
print("Model loaded.\n")

# ── Define images ───────────────────────────────────────────
# Update these paths to where your images are saved
IMAGE_PATHS = {
    "Image 1": "image1.png",  # warehouse floor, forks visible, empty + loaded pallets, blue tape
    "Image 2": "image2.png",  # multiple wrapped pallets inside blue tape lines
    "Image 3": "image3.png",  # stacked pallets with wrap on top, no blue tape, forks visible
}

# ── Define prompts ──────────────────────────────────────────
# Prompt 1: General scene understanding
SCENE_PROMPT = (
    "You are a vision system on an autonomous mobile robot (AMR) with forks in a warehouse. "
    "Describe what you see: How many pallets are visible? Are they loaded or empty? "
    "Is there shrink wrap or plastic film on any pallet? "
    "Are there blue tape lines on the floor? Are any humans or workers nearby?"
)

# Prompt 2: Readiness classification
READINESS_PROMPT = (
    "You are a vision system on a forklift robot. Your job is to determine if a pallet is ready to be picked up. "
    "A pallet is READY if: it is fully loaded, wrapped in shrink wrap, no workers are near it, "
    "and its fork pockets at the base are visible and unblocked. "
    "A pallet is NOT READY if: it is empty, partially loaded, unwrapped, or someone is working on it. "
    "Look at this image and classify each visible pallet as READY or NOT READY. "
    "For each pallet, explain why in one sentence."
)

# Prompt 3: Fork pocket check
FORK_PROMPT = (
    "Look at the base of the pallets in this image. "
    "Are the fork pockets (dark openings at the bottom of the pallet) visible and clear of debris? "
    "Can a forklift insert its forks safely? Describe any obstructions you see."
)

# Prompt 4: Blue tape detection
TAPE_PROMPT = (
    "Is there blue tape or blue line markings on the warehouse floor in this image? "
    "If yes, are any pallets positioned inside the blue tape boundaries?"
)

ALL_PROMPTS = {
    "Scene understanding": SCENE_PROMPT,
    "Readiness classification": READINESS_PROMPT,
    "Fork pocket check": FORK_PROMPT,
    "Blue tape detection": TAPE_PROMPT,
}

# ── Run inference ───────────────────────────────────────────
results = {}

for img_name, img_path in IMAGE_PATHS.items():
    if not os.path.exists(img_path):
        print(f"WARNING: {img_path} not found. Skipping {img_name}.\n")
        continue

    print(f"{'='*60}")
    print(f"Processing: {img_name} ({img_path})")
    print(f"{'='*60}")

    image = Image.open(img_path).convert("RGB")
    enc_image = model.encode_image(image)
    results[img_name] = {}

    for prompt_name, prompt_text in ALL_PROMPTS.items():
        print(f"\n--- {prompt_name} ---")
        start = time.time()

        answer = model.answer_question(enc_image, prompt_text, tokenizer)

        elapsed = time.time() - start
        print(f"{answer}")
        print(f"({elapsed:.1f}s)")

        results[img_name][prompt_name] = {
            "prompt": prompt_text,
            "response": answer,
            "inference_time_s": round(elapsed, 2),
        }

    print("\n")

# ── Save raw results to JSON ────────────────────────────────
output_file = "pallet_readiness_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Raw results saved to {output_file}")

# ── Print summary ───────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for img_name, prompts in results.items():
    readiness = prompts.get("Readiness classification", {}).get("response", "N/A")
    tape = prompts.get("Blue tape detection", {}).get("response", "N/A")
    print(f"\n{img_name}:")
    print(f"  Readiness: {readiness[:150]}...")
    print(f"  Blue tape: {tape[:150]}...")
