#!/usr/bin/env python
"""Interactive story generation with trained SimpleStories model."""

import sys
import torch
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from tokenizers import Tokenizer

print()
print("Trained models:")
print("    5M")
print("    11M")
MODEL_NAME = input("Select: ")

if MODEL_NAME == "5M":
    MODEL_PATH = "out/2026-01-14_07-56-48/checkpoints/model_step_18560.pt"
elif MODEL_NAME == "11M":
    MODEL_PATH = "out/2026-01-14_08-42-28/checkpoints/model_step_18560.pt"
else:
    print("Invalid entry.")
    sys.exit()

TOKENIZER_PATH = "simple_stories_train/tokenizer/simplestories-4096.json"

def main():
    print("Loading model...")
    model = Llama.from_pretrained(MODEL_PATH, MODEL_CONFIGS[MODEL_NAME])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Model loaded on {device}. Enter a story prompt (or 'quit' to exit).\n")

    while True:
        prompt = input("Prompt> ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.8,
                top_k=40,
            )

        story = tokenizer.decode(output[0].tolist())
        print(f"\n{story}\n")

if __name__ == "__main__":
    main()
