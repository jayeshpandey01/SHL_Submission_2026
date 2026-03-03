"""
Add transcription cells to grammar_scoring_fixed.ipynb
Inserts cells for Tasks 3.1-3.5 after cell-3 (setup)
"""

import json

# Read the notebook
with open('grammar_scoring_fixed.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create new cells for transcription (Tasks 3.1-3.5)

# Markdown cell explaining the transcription section
markdown_cell = {
    "cell_type": "markdown",
    "id": "transcription-intro",
    "metadata": {},
    "source": [
        "## Data Loading and Transcription (Tasks 3.1-3.5)\n",
        "\n",
        "### Task 3.1: Load train.csv and test.csv\n",
        "Load the dataset CSV files containing audio filenames and labels.\n",
        "\n",
        "### Task 3.2: Implement Whisper Transcription with Caching\n",
        "Use Whisper 'base' model for consistent transcription across train and test sets.\n",
        "Cache results to avoid re-running expensive transcription.\n",
        "\n",
        "### Task 3.3: Transcribe Training Audio (409 files)\n",
        "Transcribe all training audio files and save to train_transcribed.csv.\n",
        "\n",
        "### Task 3.4: Transcribe Test Audio (197 files)\n",
        "Transcribe all test audio files and save to test_transcribed.csv.\n",
        "\n",
        "### Task 3.5: Verify Transcription Quality\n",
        "Sample 5 random transcripts from each dataset to verify quality."
    ]
}

# Code cell for loading data (Task 3.1)
load_data_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "task-3-1-load-data",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Task 3.1: Load train.csv and test.csv\n",
        "print(\"=\" * 60)\n",
        "print(\"Task 3.1: Loading CSV files\")\n",
        "print(\"=\" * 60)\n",
        "\n",
        "train_df = pd.read_csv(TRAIN_CSV)\n",
        "test_df = pd.read_csv(TEST_CSV)\n",
        "\n",
        "print(f\"✓ Loaded train.csv: {len(train_df)} samples\")\n",
        "print(f\"✓ Loaded test.csv: {len(test_df)} samples\")\n",
        "print(f\"\\nTrain columns: {list(train_df.columns)}\")\n",
        "print(f\"Test columns: {list(test_df.columns)}\")\n",
        "print(f\"\\nTrain data preview:\")\n",
        "print(train_df.head())\n",
        "print(f\"\\nTest data preview:\")\n",
        "print(test_df.head())\n",
        "\n",
        "# Check for label column (train has 'label', test doesn't)\n",
        "if 'label' in train_df.columns:\n",
        "    print(f\"\\nLabel statistics:\")\n",
        "    print(train_df['label'].describe())\n",
        "    print(f\"\\nLabel distribution:\")\n",
        "    print(train_df['label'].value_counts().sort_index())"
    ]
}

# Code cell for Whisper transcription (Tasks 3.2, 3.3, 3.4)
transcription_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "task-3-2-3-4-transcription",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Task 3.2: Implement Whisper transcription function with caching\n",
        "import torch\n",
        "\n",
        "# Determine device\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "print(f\"Using device: {device}\")\n",
        "\n",
        "# Load Whisper base model\n",
        "print(\"\\nLoading Whisper 'base' model...\")\n",
        "whisper_model = whisper.load_model(\"base\", device=device)\n",
        "print(\"✓ Whisper 'base' model loaded successfully\")\n",
        "\n",
        "def transcribe_audio_file(model, audio_path):\n",
        "    \"\"\"Transcribe a single audio file\"\"\"\n",
        "    try:\n",
        "        if not os.path.exists(audio_path):\n",
        "            return None\n",
        "        \n",
        "        result = model.transcribe(audio_path, language='en', fp16=(device=='cuda'))\n",
        "        return result['text'].strip()\n",
        "    except Exception as e:\n",
        "        print(f\"Error transcribing {audio_path}: {e}\")\n",
        "        return None\n",
        "\n",
        "def transcribe_dataset(model, df, audio_dir, cache_file, dataset_name):\n",
        "    \"\"\"\n",
        "    Transcribe all audio files in a dataset with caching.\n",
        "    \n",
        "    Args:\n",
        "        model: Whisper model\n",
        "        df: DataFrame with 'filename' column\n",
        "        audio_dir: Directory containing audio files\n",
        "        cache_file: Path to save/load cached transcriptions\n",
        "        dataset_name: Name for logging (e.g., 'train', 'test')\n",
        "    \n",
        "    Returns:\n",
        "        DataFrame with added 'transcription' column\n",
        "    \"\"\"\n",
        "    print(\"\\n\" + \"=\" * 60)\n",
        "    print(f\"Transcribing {dataset_name} dataset\")\n",
        "    print(\"=\" * 60)\n",
        "    \n",
        "    # Check if cached transcriptions exist\n",
        "    if os.path.exists(cache_file):\n",
        "        print(f\"✓ Found cached transcriptions at {cache_file}\")\n",
        "        cached_df = pd.read_csv(cache_file)\n",
        "        print(f\"✓ Loaded {len(cached_df)} cached transcriptions\")\n",
        "        return cached_df\n",
        "    \n",
        "    # Create a copy of the dataframe\n",
        "    df_transcribed = df.copy()\n",
        "    \n",
        "    # Transcribe each audio file\n",
        "    transcriptions = []\n",
        "    missing_files = []\n",
        "    \n",
        "    print(f\"Transcribing {len(df)} audio files...\")\n",
        "    \n",
        "    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f\"Transcribing {dataset_name}\"):\n",
        "        # Get audio filename (handle both with and without .wav extension)\n",
        "        audio_filename = row['filename']\n",
        "        if not audio_filename.endswith('.wav'):\n",
        "            audio_filename = f\"{audio_filename}.wav\"\n",
        "        \n",
        "        audio_path = os.path.join(audio_dir, audio_filename)\n",
        "        \n",
        "        # Transcribe\n",
        "        transcription = transcribe_audio_file(model, audio_path)\n",
        "        \n",
        "        if transcription is None:\n",
        "            missing_files.append(audio_filename)\n",
        "            transcription = \"\"  # Empty string for missing files\n",
        "        \n",
        "        transcriptions.append(transcription)\n",
        "    \n",
        "    # Add transcriptions to dataframe\n",
        "    df_transcribed['transcription'] = transcriptions\n",
        "    \n",
        "    # Save to cache\n",
        "    df_transcribed.to_csv(cache_file, index=False)\n",
        "    print(f\"\\n✓ Saved transcriptions to {cache_file}\")\n",
        "    \n",
        "    # Report statistics\n",
        "    non_empty = sum(1 for t in transcriptions if t)\n",
        "    empty = len(transcriptions) - non_empty\n",
        "    \n",
        "    print(f\"\\nTranscription Statistics:\")\n",
        "    print(f\"  Total files: {len(transcriptions)}\")\n",
        "    print(f\"  Successfully transcribed: {non_empty}\")\n",
        "    print(f\"  Empty/missing: {empty}\")\n",
        "    \n",
        "    if missing_files:\n",
        "        print(f\"\\nMissing files ({len(missing_files)}):\") \n",
        "        for f in missing_files[:10]:  # Show first 10\n",
        "            print(f\"  - {f}\")\n",
        "        if len(missing_files) > 10:\n",
        "            print(f\"  ... and {len(missing_files) - 10} more\")\n",
        "    \n",
        "    return df_transcribed\n",
        "\n",
        "# Task 3.3: Transcribe training audio (409 files)\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"Task 3.3: Transcribing training audio (409 files)\")\n",
        "print(\"=\" * 60)\n",
        "train_df = transcribe_dataset(\n",
        "    whisper_model, train_df, TRAIN_AUDIO_DIR, \n",
        "    TRAIN_TRANSCRIBED_CACHE, \"train\"\n",
        ")\n",
        "\n",
        "# Task 3.4: Transcribe test audio (197 files)\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"Task 3.4: Transcribing test audio (197 files)\")\n",
        "print(\"=\" * 60)\n",
        "test_df = transcribe_dataset(\n",
        "    whisper_model, test_df, TEST_AUDIO_DIR, \n",
        "    TEST_TRANSCRIBED_CACHE, \"test\"\n",
        ")"
    ]
}

# Code cell for quality verification (Task 3.5)
verification_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "task-3-5-verification",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Task 3.5: Verify transcription quality (sample 5 random transcripts)\n",
        "\n",
        "def verify_transcription_quality(df, dataset_name, n_samples=5):\n",
        "    \"\"\"Sample and display random transcriptions for quality verification\"\"\"\n",
        "    print(\"\\n\" + \"=\" * 60)\n",
        "    print(f\"Task 3.5: Verifying transcription quality for {dataset_name}\")\n",
        "    print(\"=\" * 60)\n",
        "    \n",
        "    # Filter to non-empty transcriptions\n",
        "    non_empty_df = df[df['transcription'].str.len() > 0]\n",
        "    \n",
        "    if len(non_empty_df) == 0:\n",
        "        print(f\"WARNING: No non-empty transcriptions found in {dataset_name} dataset!\")\n",
        "        return\n",
        "    \n",
        "    # Sample random transcripts\n",
        "    sample_df = non_empty_df.sample(n=min(n_samples, len(non_empty_df)), random_state=42)\n",
        "    \n",
        "    print(f\"\\nRandom sample of {len(sample_df)} transcriptions:\\n\")\n",
        "    \n",
        "    for idx, row in sample_df.iterrows():\n",
        "        audio_file = row['filename']\n",
        "        transcription = row['transcription']\n",
        "        label = row.get('label', 'N/A')\n",
        "        \n",
        "        print(f\"File: {audio_file}\")\n",
        "        print(f\"Label: {label}\")\n",
        "        print(f\"Transcription: {transcription}\")\n",
        "        print(f\"Length: {len(transcription)} chars, {len(transcription.split())} words\")\n",
        "        print(\"-\" * 60)\n",
        "    \n",
        "    # Overall statistics\n",
        "    transcription_lengths = non_empty_df['transcription'].str.len()\n",
        "    word_counts = non_empty_df['transcription'].str.split().str.len()\n",
        "    \n",
        "    print(f\"\\nOverall Statistics for {dataset_name}:\")\n",
        "    print(f\"  Total transcriptions: {len(df)}\")\n",
        "    print(f\"  Non-empty transcriptions: {len(non_empty_df)}\")\n",
        "    print(f\"  Empty transcriptions: {len(df) - len(non_empty_df)}\")\n",
        "    print(f\"  Avg transcription length: {transcription_lengths.mean():.1f} chars\")\n",
        "    print(f\"  Avg word count: {word_counts.mean():.1f} words\")\n",
        "    print(f\"  Min word count: {word_counts.min()}\")\n",
        "    print(f\"  Max word count: {word_counts.max()}\")\n",
        "    print(f\"  Median word count: {word_counts.median():.1f}\")\n",
        "\n",
        "# Verify train transcriptions\n",
        "verify_transcription_quality(train_df, \"train\", n_samples=5)\n",
        "\n",
        "# Verify test transcriptions\n",
        "verify_transcription_quality(test_df, \"test\", n_samples=5)\n",
        "\n",
        "print(\"\\n\" + \"=\" * 60)\n",
        "print(\"✓ ALL TRANSCRIPTION TASKS COMPLETED SUCCESSFULLY\")\n",
        "print(\"=\" * 60)\n",
        "print(f\"\\nOutput files:\")\n",
        "print(f\"  - {TRAIN_TRANSCRIBED_CACHE}\")\n",
        "print(f\"  - {TEST_TRANSCRIBED_CACHE}\")\n",
        "print(\"\\nThese files contain the original data plus a 'transcription' column.\")\n",
        "print(\"You can now proceed with feature extraction.\")"
    ]
}

# Find the index to insert after cell-3
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell.get('id') == 'cell-3':
        insert_index = i + 1
        break

if insert_index is None:
    print("ERROR: Could not find cell-3 in notebook")
    exit(1)

# Insert the new cells
notebook['cells'].insert(insert_index, markdown_cell)
notebook['cells'].insert(insert_index + 1, load_data_cell)
notebook['cells'].insert(insert_index + 2, transcription_cell)
notebook['cells'].insert(insert_index + 3, verification_cell)

# Save the modified notebook
with open('grammar_scoring_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✓ Successfully added transcription cells to grammar_scoring_fixed.ipynb")
print(f"  - Inserted 4 new cells after cell-3 (at index {insert_index})")
print(f"  - Total cells now: {len(notebook['cells'])}")
print("\nNew cells:")
print("  1. Markdown: Transcription introduction (Tasks 3.1-3.5)")
print("  2. Code: Task 3.1 - Load train.csv and test.csv")
print("  3. Code: Tasks 3.2-3.4 - Whisper transcription with caching")
print("  4. Code: Task 3.5 - Verify transcription quality")
print("\nYou can now open grammar_scoring_fixed.ipynb in Jupyter and run these cells.")
