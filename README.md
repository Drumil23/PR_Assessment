# Pallet Readiness Vision Language Assessment

## Overview
This project evaluates pallet pickup readiness from warehouse images using a local vision-language model (VLM). The script runs a structured prompt suite over three images and records model outputs and inference latency for each task.

## Assessment Goal
Determine whether visible pallets are ready for autonomous forklift pickup using four checks:
- Scene understanding
- Readiness classification (READY / NOT READY)
- Fork pocket visibility and clearance
- Floor boundary (blue tape) detection

## Repository Structure
- `pallet_readiness_vlm.py`: Main evaluation script
- `images/`: Input images (`img1.png`, `img2.png`, `img3.png`)
- `pallet_readiness_results.json`: Generated raw inference output

## Technical Approach
The script:
1. Encodes each image to base64
2. Sends non-streaming inference requests to Ollama (`/api/generate`)
3. Executes four prompts per image
4. Stores prompt, response, and inference time (seconds) in JSON
5. Prints a short terminal summary

## Requirements
- Python 3.9+
- Ollama running locally at `http://localhost:11434`
- LLaVA model pulled in Ollama
- Python packages:
  - `requests`

## Setup
```bash
pip install requests
ollama pull llava
ollama serve
```

## Run
```bash
python pallet_readiness_vlm.py
```

## Output
- `pallet_readiness_results.json` with nested results by image and prompt:
  - `prompt`
  - `response`
  - `inference_time_s`

## Notes
- Results are model-dependent and may vary run-to-run.
- In this baseline, outputs are captured as raw model text without post-processing or confidence calibration.
- Prompt engineering and output validation can improve reliability for production robotics workflows.
