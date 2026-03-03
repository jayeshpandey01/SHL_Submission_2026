# Grammar Scoring Engine - Submission

## Overview

This project implements a grammar scoring system for SHL's 2025 intern assessment. The system predicts continuous grammar MOS (Mean Opinion Score) scores from 0-5 based on spoken English responses.

## Problem Statement

The original model suffered from severe overfitting:
- Cross-Validation RMSE: 0.34-0.42
- Kaggle Test RMSE: 0.89
- Generalization Gap: 0.55+

## Solution Approach

### Key Fixes
1. Feature Reduction: 5384 features -> 35 grammar-focused features
2. Model Simplification: 4-way stacked ensemble -> 2-model average
3. Removed Arbitrary Calibration: No manual scaling factors
4. Grammar-Specific Features: LanguageTool errors, readability metrics, disfluency markers

### Feature Engineering (~35 features)
- Grammar Error Features (LanguageTool): Total errors, error density, spelling/grammar/style/punctuation errors
- Readability Metrics (textstat): Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG Index, etc.
- Disfluency Markers: Filler words, repetitions, incomplete sentences, vocabulary diversity
- Syntactic Complexity: Sentence length distribution, punctuation patterns, capitalization consistency

### Model Strategy
- Ridge Regression (alpha=20): Strong regularization linear baseline
- LightGBM (max_depth=3, n_estimators=75): Conservative non-linear model
- Simple Ensemble: 50/50 average of both models (no stacking, no calibration)
- Prediction Clipping: All predictions clipped to [0, 5] range

## Repository Structure

```
grammar-scoring-engine/
├── SHL_GrammarScoreCheckerModel.ipynb      # Original notebook (baseline)
├── grammar_scoring_fixed.ipynb             # Fixed version with grammar features
├── grammar_scoring_improved.ipynb          # Improved version
├── transcribe_audio.py                     # Audio transcription script
├── add_transcription_cells.py              # Notebook cell generator
├── submission.csv                          # Final submission (filename, label)
├── submission_top10_final.csv              # Alternative submission format
├── dataset/
│   ├── csvs/
│   │   ├── train.csv                       # Training labels
│   │   └── test.csv                        # Test filenames
│   └── audios/
│       ├── train/                          # 409 training audio files
│       └── test/                           # 197 test audio files
└── SUBMISSION_README.md                    # This file
```

## Usage

### Running the Fixed Notebook

1. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper language-tool-python textstat lightgbm
```

2. Run on Kaggle (recommended):
   - Fork the notebook
   - Add dataset as input
   - Enable GPU
   - Run all cells

3. Run locally:
```bash
jupyter notebook grammar_scoring_fixed.ipynb
```

### Using Transcription Script

```bash
python transcribe_audio.py
```

This will:
- Load train.csv and test.csv
- Transcribe all audio files using Whisper base model
- Cache results to train_transcribed.csv and test_transcribed.csv

## Submission Format

The submission file must have exactly 2 columns:
- filename: Audio filename (without .wav extension)
- label: Predicted grammar score (0-5 range)

Example:
```csv
filename,label
audio_141,2.68
audio_114,3.23
audio_17,2.83
```

## Performance Metrics

### Target Metrics
- CV RMSE: 0.45-0.55
- Kaggle Test RMSE: 0.50-0.60
- Generalization Gap: <0.10
- Fold Consistency: std <0.08
- Feature Count: 35
- Feature-Sample Ratio: 0.086

### Key Improvements
- 99.4% reduction in feature dimensionality
- 33-44% improvement in test RMSE (expected)
- 82% reduction in generalization gap (expected)
- 50% simpler model architecture

## Technical Details

### Feature-Sample Ratio
- Training samples: 409
- Features: 35
- Ratio: 0.086 (should be <0.1 for healthy generalization)

### Cross-Validation
- 5-fold CV with shuffle=True, random_state=42
- Fold consistency target: std <0.1
- Stratified sampling for extreme score bins

### Error Handling
- Empty transcriptions handled gracefully
- Missing audio files return empty string
- Predictions clipped to [0, 5] range

## Files Reference

| File | Purpose |
|------|---------|
| SHL_GrammarScoreCheckerModel.ipynb | Original baseline implementation |
| grammar_scoring_fixed.ipynb | Fixed version with grammar features |
| grammar_scoring_improved.ipynb | Improved version with better regularization |
| transcribe_audio.py | Standalone transcription script |
| submission.csv | Final submission file |
| dataset/csvs/train.csv | Training data with labels |
| dataset/csvs/test.csv | Test data (filenames only) |

