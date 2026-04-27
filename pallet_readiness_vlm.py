import requests
import base64
import json
import time
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava"

# Defining images 
IMAGE_PATHS = {
    "Image 1": "images/img1.png",
    "Image 2": "images/img2.png",
    "Image 3": "images/img3.png",
}

# Defining prompts for different aspects of pallet readiness detection
PROMPTS = {
    "Scene understanding": (
        "You are a vision system on an autonomous mobile robot (AMR) with forks "
        "in a warehouse. Describe what you see in detail: How many pallets are "
        "visible? Are they loaded or empty? Is there shrink wrap or plastic film "
        "on any pallet? Are there blue tape lines on the floor? Are any humans "
        "or workers nearby? Describe the overall warehouse environment."
    ),
    "Readiness classification": (
        "You are a vision system on a forklift robot. Your job is to determine "
        "if a pallet is ready to be picked up and moved.\n\n"
        "A pallet is READY if: it is fully loaded, wrapped in shrink wrap, "
        "no workers are actively near it, and its fork pockets at the base "
        "are visible and unblocked.\n\n"
        "A pallet is NOT READY if: it is empty, partially loaded, unwrapped, "
        "or someone is actively working on it.\n\n"
        "Look at this image and classify each visible pallet as READY or NOT READY. "
        "For each pallet, explain your reasoning in one sentence."
    ),
    "Fork pocket check": (
        "Look carefully at the base of the pallets in this image. "
        "Are the fork pockets (the dark rectangular openings at the bottom "
        "of the pallet where a forklift inserts its forks) visible? "
        "Are they clear of debris and obstructions? "
        "Can a forklift safely insert its forks? Describe what you see at the base."
    ),
    "Blue tape detection": (
        "Look at the warehouse floor in this image carefully. "
        "Is there blue tape or blue line markings on the floor? "
        "If yes, describe where the blue lines are and whether any pallets "
        "are positioned inside the blue tape boundaries. "
        "If no blue lines are visible, state that clearly."
    ),
}

def encode_image(image_path):
    """Convert image to base64 string for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Run inference on each image with each prompt and store results
results = {}

for img_name, img_path in IMAGE_PATHS.items():
    if not os.path.exists(img_path):
        print(f"WARNING: {img_path} not found. Skipping {img_name}.\n")
        continue

    print(f"Processing: {img_name} ({img_path})")

    img_b64 = encode_image(img_path)
    results[img_name] = {}

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n--- {prompt_name} ---")
        start = time.time()

        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt_text,
            "images": [img_b64],
            "stream": False,
        })

        answer = resp.json().get("response", "No response")
        elapsed = time.time() - start

        print(answer)
        print(f"\n({elapsed:.1f}s)")

        results[img_name][prompt_name] = {
            "prompt": prompt_text,
            "response": answer,
            "inference_time_s": round(elapsed, 2),
        }

    print("\n")

# Save raw results to JSON file for further analysis
output_file = "pallet_readiness_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nRaw results saved to {output_file}")

# Print summary of key findings
print("SUMMARY")
for img_name, data in results.items():
    readiness = data.get("Readiness classification", {}).get("response", "N/A")
    tape = data.get("Blue tape detection", {}).get("response", "N/A")
    print(f"\n{img_name}:")
    print(f"  Readiness: {readiness[:200]}...")
    print(f"  Blue tape: {tape[:200]}...")