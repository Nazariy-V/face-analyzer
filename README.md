```markdown
# Face Analyzer (prototype)

This prototype analyzes a user's face (front + side) from webcam or images and extracts:

- face shape (e.g. oval / round / square / heart / oblong / diamond)
- eye shape (round / almond / hooded) and eye color
- hair length (short / medium / long) and hair color

Outputs are saved as JSON and the recorder can save video segments for later reprocessing.

Highlights (what this repo implements)
- PyTorch (`face-alignment`) landmark-based analysis with fallbacks for robustness
- Deferred processing: record front and side segments first, analyze later to avoid blocking
- Image-quality filtering: blur (Laplacian variance), brightness, contrast, roll angle and minimum landmarks
- Live preview overlay during recording with quick quality metrics and occasional landmark overlays
- Auto-relax reprocessing: if detection rate is too low the tool can progressively relax thresholds and re-run analysis
- FFmpeg-based video writer (stdin streaming) with OpenCV fallback; optional NVENC preference

Requirements

- Python 3.8+ (64-bit recommended)
- See `requirements.txt` for Python dependencies. Install with:

```powershell
pip install -r requirements.txt
```

Quick usage

- Capture two still frames via webcam (press SPACE to capture front, then press Enter to start side capture):

```powershell
python main.py --capture --output results.json
```

- Analyze two existing images:

```powershell
python main.py --front front.jpg --side side.jpg --output results.json
```

- Record longer segments (front then side) for dataset collection and processing:

```powershell
python main.py --record --duration 10 --fps 15 --record-dir .\outputs\session1
```

This will save `front.mp4`, `side.mp4`, `results.json` and `final.json` inside the record folder. Use `--sample-every N` to process every Nth frame.

FFmpeg notes (higher quality, lower CPU)

- The recorder now prefers writing videos with `ffmpeg` via stdin streaming for higher-quality H.264 encoding and optional NVENC hardware acceleration.
- CLI flags added:
	- `--no-ffmpeg` : force use of OpenCV writer (no ffmpeg)
	- `--ffmpeg-crf <int>` : CRF for libx264 (default 18; lower = better quality)
	- `--ffmpeg-preset <str>` : libx264 preset (default `veryfast`)
	- `--use-nvenc` : prefer NVIDIA NVENC H.264 encoder if available

- Make sure `ffmpeg.exe` is on your PATH (or place it in a folder that is on PATH). Example path from this environment:
	`C:\Users\Nazariy\Downloads\ffmpeg-8.0\ffmpeg-8.0\bin\ffmpeg.exe`

Example (use NVENC if available):

```powershell
python main.py --record --duration 6 --use-nvenc
```

If ffmpeg is missing or fails, the code falls back to the built-in OpenCV VideoWriter automatically.

GPU (PyTorch) setup
--------------------

The repo supports an accelerated PyTorch backend (used by `face-alignment`) when `torch` + CUDA are available.

CPU-only example:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install face-alignment
```

Common CUDA wheel examples (pick the right tag for your system):

```powershell
python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install face-alignment
```

After installing, verify with:

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
```

The program auto-detects `torch` + CUDA and uses the GPU backend when available.

Design notes and caveats
-----------------------

- This is a heuristic research prototype. Results are probabilistic and may be wrong on many inputs.
- Face shape, eye shape and hair heuristics are implemented from landmarks and simple color heuristics; for production, consider training classifiers or using specialized segmentation and color analysis models.
- Some wheels (MediaPipe etc.) can be tricky on very new Python versions; the code prefers `face-alignment` (PyTorch) for more predictable installs.

Next steps (optional)
---------------------

- Add `--ffmpeg-path` to point to the ffmpeg binary explicitly (handy if not on PATH)
- Implement a streaming producer/consumer writer thread so frames are encoded while captured (reduces memory use)
- Add an NVENC capability check on startup (run `ffmpeg -encoders` and look for `h264_nvenc`)
- Add unit tests for `process_frames`, `compute_summary`, and the writer routine

Contact / Contributions
-----------------------

Open an issue or PR in this repository with improvements, encoder options, or better heuristics.