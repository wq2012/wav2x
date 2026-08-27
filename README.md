# wav2x

[![PyPI](https://img.shields.io/pypi/v/wav2x.svg)](https://pypi.org/project/wav2x/)
[![Build Status](https://github.com/wq2012/wav2x/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/wq2012/wav2x/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A lightweight Python package for audio representation learning, spoken language identification, speaker recognition (d-vector), and voice activity detection (VAD) with TensorFlow Lite.

`wav2x` replaces the heavy C++ Lingvo dependency with a pure, standalone TensorFlow/NumPy frontend that runs on standard Python (including Python 3.10 and 3.11+) across macOS, Linux, and Windows.

---

## Installation

Install the package from PyPI:

```bash
pip install wav2x
```

Or install from the repository in development mode:

```bash
git clone https://github.com/wq2012/wav2x.git
cd wav2x
pip install -r requirements.txt
pip install -e .
```

---

## Features

- **Lingvo-compatible Log-Mel Spectrogram Frontend**: High-fidelity pure TensorFlow reimplementation of Lingvo's `MelAsrFrontend` (framing, pre-emphasis, windowing, overdrive FFT, Mel filterbank, log compression, frame stacking, and subsampling).
- **Spoken Language Identification (Lang-ID)**: Conformer-based streaming model classifying speech into 100+ supported languages.
- **Speaker Recognition (Speaker-ID / d-vector)**: Conformer-based speaker encoder producing robust d-vector representations for cross-lingual speaker verification and identification.
- **Integrated Voice Activity Detection (VAD)**: Multi-stage neural VAD for speech frame filtering prior to encoder inference.
- **Convenient HuggingFace Hub Integration**: Auto-download pretrained TFLite models on the fly via `from_pretrained()`.

---

## Quickstart & Usage

### 1. Spoken Language Identification

Identify the spoken language of an audio file using the Conformer Lang-ID model:

```python
from wav2x import WavToLangRunner

# Automatically downloads models from HuggingFace Hub if not present locally
runner = WavToLangRunner.from_pretrained(model_dir="models")

# Predict language
language_code, probs = runner.wav_to_lang("speech.wav")
print(f"Predicted language: {language_code}")
```

You can also pass explicit model paths:

```python
runner = WavToLangRunner(
    vad_model_file="models/vad_short_model.tflite",
    vad_mean_stddev_file="models/vad_short_mean_stddev.csv",
    langid_model_file="models/conformer_langid_medium.tflite",
    vad_threshold=0.1,
)
language_code, probs = runner.wav_to_lang("speech.wav")
```

---

### 2. Speaker Recognition (d-vector & Speaker Verification)

Compute d-vector embeddings and speaker verification similarity scores:

```python
from wav2x import WavToDvectorRunner

runner = WavToDvectorRunner.from_pretrained(model_dir="models")

# Compute d-vector sequence for an audio file
dvectors = runner.wav_to_dvector("utterance1.wav")
last_dvector = dvectors[-1, :]

# Compute cosine similarity between enrolled audio and test audio
similarity = runner.compute_score(
    enroll_audio_list=["enroll1.wav", "enroll2.wav"],
    test_audio="test.wav"
)
print(f"Speaker similarity score: {similarity:.4f}")
```

---

### 3. Speaker Enrollment & Identification

Enroll multiple known speakers and identify speakers in test audio:

```python
from wav2x import WavToDvectorRunner

runner = WavToDvectorRunner.from_pretrained(model_dir="models")

# Enroll speakers
runner.enroll_speaker("Alice", ["alice_audio_1.wav", "alice_audio_2.wav"])
runner.enroll_speaker("Bob", ["bob_audio_1.wav"])

# Identify speaker in unknown audio
speaker_name, score = runner.identify_speaker("unknown.wav", threshold=0.50)
if speaker_name:
    print(f"Identified as {speaker_name} with score {score:.4f}")
else:
    print(f"Unknown speaker (best score: {score:.4f})")
```

---

### 4. Audio Feature Extraction (Log-Mel Spectrogram)

Extract the exact 512-dimensional stacked log-mel features used by Conformer models:

```python
from wav2x import LogMelFeatureExtractor
import soundfile as sf

extractor = LogMelFeatureExtractor(
    frame_size_ms=32.0,
    frame_step_ms=10.0,
    num_bins=128,
    sample_rate=16000.0,
    stack_left_context=3,
    frame_stride=3,
)

# audio samples shaped [1, time] with int16 range [-32768, 32767]
data, sr = sf.read("audio.wav")
samples = (data * 32768.0).astype("int16").reshape(1, -1)

# features shape: [1, num_frames, 512]
features = extractor.extract(samples)
print(f"Extracted feature shape: {features.shape}")
```

---

## Interactive Web Demos

Interactive Gradio web interfaces are available under `demos/`:

```bash
pip install gradio
python demos/lang-id-demo.py
python demos/speaker-id-demo.py
```

---

## Running Tests

Run the full unit and end-to-end test suite:

```bash
bash run_tests.sh
```

Linting:

```bash
flake8 --indent-size 2 --max-line-length 80 .
```

---

## Models

Pretrained models are hosted on Hugging Face Hub:
- **Language Identification**: [`tflite-hub/conformer-lang-id`](https://huggingface.co/tflite-hub/conformer-lang-id)
- **Speaker Encoder**: [`tflite-hub/conformer-speaker-encoder`](https://huggingface.co/tflite-hub/conformer-speaker-encoder)

---

## Citations

The underlying conformer architectures and training methodologies are described in:

```bibtex

@inproceedings{pelecanos2022parameter,
  title={Parameter-Free Attentive Scoring for Speaker Verification},
  author={Jason Pelecanos and Quan Wang and Yiling Huang and Ignacio Lopez Moreno},
  booktitle={Odyssey: The Speaker and Language Recognition Workshop},
  year={2022}
}

@inproceedings{wang2022attentive,
  title={Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech},
  author={Quan Wang and Yang Yu and Jason Pelecanos and Yiling Huang and Ignacio Lopez Moreno},
  booktitle={Odyssey: The Speaker and Language Recognition Workshop},
  year={2022}
}

@inproceedings{chojnacka2021speakerstew,
  title={{SpeakerStew: Scaling to many languages with a triaged multilingual text-dependent and text-independent speaker verification system}},
  author={Chojnacka, Roza and Pelecanos, Jason and Wang, Quan and Moreno, Ignacio Lopez},
  booktitle={Prod. Interspeech},
  pages={1064--1068},
  year={2021},
  doi={10.21437/Interspeech.2021-646},
  issn={2958-1796},
}
```

---

## License

Apache License 2.0
