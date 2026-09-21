# TrOCR Training Data Pipeline & Token Structure Audit Report

**Date**: September 2026  
**Repository Branch**: `person-B`  
**Inspected Manifests**: 
- `training/datasets/kannada_character_balanced_v2/train_v2.jsonl` (11,179 samples)
- `training/datasets/kannada_character_balanced_v2/validation_v2.jsonl`
- Source: `IIIT-INDIC-HW-WORDS`

---

## 1. Executive Diagnosis

The reported symptom of TrOCR emitting **"a run-on character blob with zero inter-word spacing"** is directly caused by the underlying training dataset tokenization:

1. **All Training Samples are Isolated Single Words**:
   Inspection of `train_v2.jsonl` demonstrates that all 11,179 training pairs consist of cropped single words without spaces:
   - Example 1: `{"image": ".../69053.jpg", "text": "ಸರಿಸಮಾನಗಿ"}`
   - Example 2: `{"image": ".../50266.jpg", "text": "ಸಾರ್ವಜನಿಕವಾಗಿ"}`
   - Example 3: `{"image": ".../34798.jpg", "text": "ಹೆಚ್ಚುವರಿಯಾಗಿ"}`
2. **Missing Inter-Word Space Token Distribution**:
   Because TrOCR (Vision-Encoder-Decoder) was fine-tuned with `max_sequence_length: 32` exclusively on single-word crops, its autoregressive decoder learned zero probability transitions for space tokens between word boundaries.
3. **Behavior on Document Lines**:
   When passed a multi-word document line crop, TrOCR attempts to decode the whole image as a single continuous lexical sequence, outputting concatenated character blobs.

---

## 2. Solution Architecture

To solve this systematically without corrupting the existing single-word recognition capabilities:

1. **Upstream Text Detection**:
   PaddleOCR's DBNet text detector (`det=True`) is deployed upstream of recognition. It detects word/token bounding boxes across both printed and handwritten regions.
2. **Geometric Spacing Reconstruction**:
   Horizontal spatial gaps $\Delta x = x_{i+1}^{\min} - x_i^{\max}$ between consecutive word boxes on the same baseline are evaluated. If $\Delta x > \tau \cdot \text{line\_height}$, an explicit whitespace is inserted.
3. **Line-Level Training Pipeline (`generate_synthetic_lines.py`)**:
   A dedicated synthetic data generator is provided to synthesize full-sentence and tabular line crops with varied spacing, Kannada handwriting fonts, and realistic scan degradation (blur, skew, ink bleed).
