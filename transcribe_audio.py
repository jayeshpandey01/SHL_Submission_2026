"""
Audio Transcription Script for Grammar Scoring
Tasks 3.1-3.5: Load CSVs, transcribe audio with Whisper base model, cache results
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
TRAIN_CSV = 'dataset/csvs/train.csv'
TEST_CSV = 'dataset/csvs/test.csv'
TRAIN_AUDIO_DIR = 'dataset/audios/train'
TEST_AUDIO_DIR = 'dataset/audios/test'
TRAIN_TRANSCRIBED_CSV = 'dataset/csvs/train_transcribed.csv'
TEST_TRANSCRIBED_CSV = 'dataset/csvs/test_transcribed.csv'

def load_csvs():
    """Task 3.1: Load train.csv and test.csv"""
    print("=" * 60)
    print("Task 3.1: Loading CSV files")
    print("=" * 60)
    
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"✓ Loaded train.csv: {len(train_df)} samples")
    print(f"✓ Loaded test.csv: {len(test_df)} samples")
    print(f"\nTrain columns: {list(train_df.columns)}")
    print(f"Test columns: {list(test_df.columns)}")
    
    return train_df, test_df

def load_whisper_model():
    """Load Whisper base model"""
    print("\n" + "=" * 60)
    print("Loading Whisper base model...")
    print("=" * 60)
    
    # Import whisper here to avoid DLL issues at module load time
    import whisper
    
    model = whisper.load_model("base")
    print("✓ Whisper base model loaded successfully")
    
    return model

def transcribe_audio_file(model, audio_path):
    """Transcribe a single audio file"""
    try:
        if not os.path.exists(audio_path):
            return None
        
        result = model.transcribe(audio_path, language='en')
        return result['text'].strip()
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None

def transcribe_dataset(model, df, audio_dir, output_csv, dataset_name):
    """
    Task 3.2: Implement Whisper transcription function with caching
    Task 3.3/3.4: Transcribe all audio files
    """
    print("\n" + "=" * 60)
    print(f"Transcribing {dataset_name} dataset")
    print("=" * 60)
    
    # Check if cached transcriptions exist
    if os.path.exists(output_csv):
        print(f"✓ Found cached transcriptions at {output_csv}")
        cached_df = pd.read_csv(output_csv)
        print(f"✓ Loaded {len(cached_df)} cached transcriptions")
        return cached_df
    
    # Create a copy of the dataframe
    df_transcribed = df.copy()
    
    # Add transcription column if it doesn't exist
    if 'transcription' not in df_transcribed.columns:
        df_transcribed['transcription'] = None
    
    # Transcribe each audio file
    transcriptions = []
    missing_files = []
    
    print(f"Transcribing {len(df)} audio files...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Transcribing {dataset_name}"):
        # Get audio filename
        audio_filename = row['audio_filename']
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Transcribe
        transcription = transcribe_audio_file(model, audio_path)
        
        if transcription is None:
            missing_files.append(audio_filename)
            transcription = ""  # Empty string for missing files
        
        transcriptions.append(transcription)
    
    # Add transcriptions to dataframe
    df_transcribed['transcription'] = transcriptions
    
    # Save to cache
    df_transcribed.to_csv(output_csv, index=False)
    print(f"\n✓ Saved transcriptions to {output_csv}")
    
    # Report statistics
    non_empty = sum(1 for t in transcriptions if t)
    empty = len(transcriptions) - non_empty
    
    print(f"\nTranscription Statistics:")
    print(f"  Total files: {len(transcriptions)}")
    print(f"  Successfully transcribed: {non_empty}")
    print(f"  Empty/missing: {empty}")
    
    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        for f in missing_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return df_transcribed

def verify_transcription_quality(df, dataset_name, n_samples=5):
    """Task 3.5: Verify transcription quality (sample 5 random transcripts)"""
    print("\n" + "=" * 60)
    print(f"Task 3.5: Verifying transcription quality for {dataset_name}")
    print("=" * 60)
    
    # Sample random transcripts
    sample_df = df[df['transcription'].str.len() > 0].sample(n=min(n_samples, len(df)), random_state=42)
    
    print(f"\nRandom sample of {len(sample_df)} transcriptions:\n")
    
    for idx, row in sample_df.iterrows():
        audio_file = row['audio_filename']
        transcription = row['transcription']
        score = row.get('score', 'N/A')
        
        print(f"File: {audio_file}")
        print(f"Score: {score}")
        print(f"Transcription: {transcription}")
        print(f"Length: {len(transcription)} chars, {len(transcription.split())} words")
        print("-" * 60)
    
    # Overall statistics
    transcription_lengths = df[df['transcription'].str.len() > 0]['transcription'].str.len()
    word_counts = df[df['transcription'].str.len() > 0]['transcription'].str.split().str.len()
    
    print(f"\nOverall Statistics:")
    print(f"  Avg transcription length: {transcription_lengths.mean():.1f} chars")
    print(f"  Avg word count: {word_counts.mean():.1f} words")
    print(f"  Min word count: {word_counts.min()}")
    print(f"  Max word count: {word_counts.max()}")

def main():
    """Execute tasks 3.1-3.5"""
    print("\n" + "=" * 60)
    print("GRAMMAR SCORING: AUDIO TRANSCRIPTION PIPELINE")
    print("Tasks 3.1-3.5")
    print("=" * 60)
    
    # Task 3.1: Load CSVs
    train_df, test_df = load_csvs()
    
    # Task 3.2: Load Whisper model
    model = load_whisper_model()
    
    # Task 3.3: Transcribe training audio (409 files)
    print("\n" + "=" * 60)
    print("Task 3.3: Transcribing training audio (409 files)")
    print("=" * 60)
    train_transcribed = transcribe_dataset(
        model, train_df, TRAIN_AUDIO_DIR, TRAIN_TRANSCRIBED_CSV, "train"
    )
    
    # Task 3.4: Transcribe test audio (197 files)
    print("\n" + "=" * 60)
    print("Task 3.4: Transcribing test audio (197 files)")
    print("=" * 60)
    test_transcribed = transcribe_dataset(
        model, test_df, TEST_AUDIO_DIR, TEST_TRANSCRIBED_CSV, "test"
    )
    
    # Task 3.5: Verify transcription quality
    verify_transcription_quality(train_transcribed, "train", n_samples=5)
    verify_transcription_quality(test_transcribed, "test", n_samples=5)
    
    print("\n" + "=" * 60)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {TRAIN_TRANSCRIBED_CSV}")
    print(f"  - {TEST_TRANSCRIBED_CSV}")
    print("\nThese files contain the original data plus a 'transcription' column.")

if __name__ == "__main__":
    main()
