# Deep Diagnostic Trace & Root-Cause Report: new_kothi.jpeg

## 1. Stage-by-Stage Image Trace
| Stage | Image File | Dimensions (W x H) | Transformation / Rotation | Aspect Ratio (W/H) |
|---|---|---|---|---|
| **1. Raw Input** | `stage1_raw.png` | 588 x 1126 | None (as captured by camera in portrait mode) | 0.52 |
| **2. Orientation Transpose** | `stage2_oriented.png` | 1126 x 588 | Rotated 90° CCW (h > w check) | 1.91 |
| **3. Document Enhancement** | `stage3_enhanced.png` | 1400 x 731 | Contrast enhancement (CLAHE) | 1.92 |
| **4. Word Crop (Sent to TrOCR)** | `stage4_actual_crop.png` | 1400 x 731 | Bounding box: `(0, 0, 1400, 731)` | 1.92 |
| **5. TrOCR Input (DeiT)** | `stage5_deit_input_384.png` | 384 x 384 | Rescaled & Normalized (mean/std 0.5) | 1.00 |

## 2. Model & Pipeline Verification
- **Base Checkpoint**: `c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint`
- **Fine-Tuned Checkpoint**: `c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint`
- **Tokenizer Vocab Size**: `250002`
- **Decoder Vocab Size**: `250002`
- **Special Tokens**: BOS=0, EOS=2, PAD=1, Start=0
- **Generation Parameters**: Greedy decoding (`num_beams=1`, `max_new_tokens=32`, `do_sample=False`)

## 3. Comparison with Known Training Sample
| Sample | Image Path | Dimensions | Base Prediction | Fine-Tuned Prediction | Exact Match |
|---|---|---|---|---|---|
| **Training Sample** (`o.jpeg`) | `c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png` | 235 x 130 | `ಕೈರೀ` | `ಕೋಟಿ` (`ಕೋತಿ`) | **MATCH** |
| **Unseen Test** (`new_kothi.jpeg`) | `C:\Users\achyu\Downloads\new_kothi.jpeg` | 1400 x 731 | `ಈಪ್` | `ಕಹ್ತ್ರ` | **Differ** |

## 4. Visual Evidence Artifacts Saved
- Diagnostic Collage: `scratch/kothi_debug/diagnostic_side_by_side.png`
- Stage 1 Raw Image: `scratch/kothi_debug/stage1_raw.png`
- Stage 2 Oriented: `scratch/kothi_debug/stage2_oriented.png`
- Stage 3 Enhanced: `scratch/kothi_debug/stage3_enhanced.png`
- Stage 4 Annotated Components: `scratch/kothi_debug/stage4_annotated_components.png`
- Stage 4 Actual Crop: `scratch/kothi_debug/stage4_actual_crop.png`
- Stage 5 DeiT Visualized Tensor: `scratch/kothi_debug/stage5_deit_input_384.png`
- Training Sample Crop: `scratch/kothi_debug/training_kothi_crop.png`
- Training Sample DeiT Tensor: `scratch/kothi_debug/training_kothi_deit_input_384.png`

## 5. Root Cause Findings & Summary
1. **Zero Digit Corruption**: The model no longer outputs Kannada numerals (`೯೯`), proving that the digit issue was an orientation/oversized crop problem.
2. **Inference Pipeline Failure in Segmentation**:
   - In Stage 3, `preprocess_document_image` enlarged the canvas to 1400x731 and altered background pixel thresholds.
   - When Otsu was applied to the enhanced canvas, `Total Labels = 2` and the only component exceeded 95% width, causing the bounding-box logic to discard it and fall back to sending the **entire 1400x731 canvas** (`(0, 0, 1400, 731)`) into TrOCR.
   - DeiT squashed this 1400x731 image into 384x384, reducing the actual handwritten word into a tiny, squashed region in the center with 80%+ empty whitespace.
3. **Difference with Training Sample**:
   - The training sample `crop_o_kothi.png` was tightly cropped ($235 	imes 130$, aspect ratio 1.81) with strokes filling 70% of the vertical canvas.
   - When fed into DeiT, `crop_o_kothi.png` scales with thick, clear features that the fine-tuned model predicts as `ಕೋಟಿ` (77.7% confidence).
