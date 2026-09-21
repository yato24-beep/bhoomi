import json
import sys
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

tok = AutoTokenizer.from_pretrained("Chakita/KannadaBERT")
vocab = tok.get_vocab()
inv_vocab = {v: k for k, v in vocab.items()}

# Test with some known Kannada token IDs
sample_ids = tok.encode("ಕರ್ನಾಟಕ ಭೂಮಿ")
print("Encoded 'ಕರ್ನಾಟಕ ಭೂಮಿ':", sample_ids)
decoded = tok.decode(sample_ids)
print("Decoded back:", decoded)

# Inspect tokenizer config
with open("experiments/onnx_kannada_trocr/models/tokenizer/tokenizer.json", "r", encoding="utf-8") as f:
    tj = json.load(f)

print("Decoder config:", tj.get("decoder"))
