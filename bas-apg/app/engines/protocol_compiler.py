"""
BAS-APG — Zero-Shot Dynamic Procedure Parsing (Offline VLM)

Uses a highly quantized, tiny local LLM (e.g., Qwen 0.5B via Transformers)
running purely on CPU to parse a plain-text scientific manual into a
deterministic JSON Finite State Machine (FSM).

Usage:
    python app/engines/protocol_compiler.py --input data/manual.txt --output data/generated_procedure.json
"""

import argparse
import json
import os
import re
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    print("ERROR: transformers and torch are required for the Protocol Compiler.")
    print("Run: pip install transformers torch accelerate")
    sys.exit(1)

# Default to a tiny, fast model that can run on CPU
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """You are an expert aerospace systems engineer programming an AI Guardian.
Your job is to read a plain-text scientific manual and convert it into a deterministic JSON state machine.
You must output ONLY valid JSON, with absolutely no markdown formatting, no code blocks, and no conversational text.

The JSON MUST follow this exact schema:
{
  "procedure_name": "string",
  "steps": [
    {
      "step_id": "step_1",
      "description": "string",
      "valid_next_states": ["step_2"],
      "action_required": "string",  # e.g., "PICK", "PLACE", "POUR"
      "primary_object": "string",   # e.g., "Tweezers", "Red_Box"
      "timeout_seconds": integer
    }
  ]
}

If a step is the final step, its valid_next_states should be empty: [].
"""


def extract_json(text: str) -> str:
    """Extract JSON from the LLM's raw output in case it includes conversational text."""
    # Try to find JSON block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def compile_manual(input_file: str, output_file: str, model_id: str = DEFAULT_MODEL):
    print("=" * 60)
    print("  BAS-APG Zero-Shot Protocol Compiler")
    print("=" * 60)

    if not os.path.exists(input_file):
        print(f"ERROR: Input manual not found at {input_file}")
        sys.exit(1)

    with open(input_file, "r") as f:
        manual_text = f.read()

    print(
        f"Loading local VLM/LLM: {model_id} (This may take a moment to download on first run)..."
    )

    try:
        # Load tokenizer and model for causal LM
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    print("\nParsing manual offline...")

    # Format prompt for instruct models
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Manual:\n{manual_text}\n\nOutput only JSON:"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = generator(prompt, max_new_tokens=1024, do_sample=False, temperature=0.0)

    raw_output = outputs[0]["generated_text"][len(prompt) :]

    print("\nParsing LLM Output...")
    json_str = extract_json(raw_output)

    try:
        fsm_dict = json.loads(json_str)
        print("✅ Successfully parsed JSON!")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON. Raw output from LLM:\n{json_str}")
        print(f"Error: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(fsm_dict, f, indent=4)

    print(f"\n✅ Dynamic Procedure compiled and saved to {output_file}")
    print(
        f"The BAS-APG system can now execute '{fsm_dict.get('procedure_name')}' natively."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/manual.txt", help="Path to plain-text manual"
    )
    parser.add_argument(
        "--output",
        default="data/generated_procedure.json",
        help="Output path for FSM JSON",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    args = parser.parse_args()

    compile_manual(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
