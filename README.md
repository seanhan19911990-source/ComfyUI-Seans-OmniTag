# ComfyUI-Seans-OmniTag
<img width="467" height="417" alt="image" src="https://github.com/user-attachments/assets/8f043f41-2def-42ac-b4ec-6116e217de41" />



📦 Installation & Requirements
1. Install Custom Node
Clone this repo into your ComfyUI/custom_nodes/ folder:

Bash
git clone https://github.com/seanhan19911990-source/ComfyUI-Seans-OmniTag.git
2. Python Dependencies (Automatic)
The node will automatically handle the installation of transformers, whisper, and opencv on first run.

3. System Requirement: FFmpeg (Manual)
This node requires FFmpeg to be installed on your system.

Windows: Download from Gyan.dev, extract, and add the bin folder to your System Environment Variables (Path).

Linux: sudo apt install ffmpeg

Mac: brew install ffmpeg

Note: If you see ❌ ERROR: FFmpeg not found in ComfyUI, it means the node can't find the ffmpeg command in your system path.


🛠️ Sean's OmniTag Processor
The Ultimate All-in-One Captioning & Dataset Pipeline for ComfyUI
Sean's OmniTag is a powerhouse node designed to eliminate the friction of building high-quality datasets. Whether you are training LTX-Video, Flux, or SDXL, this node automates the most tedious parts of the process: video segmentation, high-fidelity visual captioning, and synchronized audio transcription.

🚀 Key Features
Omni-Input Support: Drop in a folder of images or a single video file—the node handles both seamlessly.

Abliterated Intelligence: Powered by the Qwen2.5-VL-7B-Abliterated model, providing unfiltered, clinical, and exhaustive descriptions without AI "safety" refusals.

Smart Segmentation: Automatically carves long videos into perfect training clips with intelligent "Segment Skipping" to maximize visual variety.

Audio-Sync Transcription: Uses OpenAI Whisper to listen to your clips and append spoken dialogue directly to your text captions.

Anti-Lazy Safety Net: Built-in logic detects if the AI gives a short or "lazy" response and automatically retries with a more aggressive generation pass.

📐 Resolution & Aspect Ratio Logic
One of the node's strongest assets is its Longest-Edge Scaling system.

Smart Resize: The node identifies the longest side of your media and scales it to your target_resolution (e.g., 768px).

Aspect Ratio Preservation: It never stretches or squashes your content. A 16:9 video stays 16:9, and a 9:16 TikTok stays 9:16.

High-Fidelity Interpolation: Uses Lanczos4 resampling to ensure that fine details—like the texture of a young woman's long wavy hair or the sparkle in striking dark eyes—are preserved for the AI to learn.

Training Ready: Perfectly prepares your data for the "Aspect Ratio Bucketing" used by Flux and LTX-Video.

⚙️ Parameter Guide
📂 Paths & Instructions
input_path: Path to your image folder or .mp4/.mkv file.

output_path: Where your pairs of .mp4/.png and .txt files will be saved.

trigger_word: Your LoRA's unique identifier (e.g., ohwx).

llm_instruction: Your prompt to the AI. Use {trigger} to automatically insert your trigger word into the description.

🖼️ Generation Settings
target_resolution: Max length of the longest side. 768 is recommended for Flux/LTX-V.

max_tokens: Control description depth (512 to 2048). Use 768+ for detailed video motion descriptions.

🎥 Video controls
video_segment_seconds: Duration of each output clip (e.g., 5.0s).

segment_skip: How many segments to skip between grabs. High values ensure a diverse dataset from a single video.

video_max_segments: Limits how many clips are pulled from one file to prevent dataset bias.

🎙️ Audio & Speech
include_audio_in_video: Keeps the original audio track in the exported clips.

append_speech_to_end: Transcribes dialogue and adds it to the .txt file (e.g., ...a playful smile. Audio: "Hello there!").

💡 Pro-Tips for Creators
For LTX-Video: Set max_tokens to 1024. This allows the AI enough "breath" to describe complex movements, which is vital for high-quality video LoRAs.

For Character Training: Use the default instruction to focus on physical traits. The Abliterated model excels at describing smooth skin, specific hair waves, and facial expressions without filter interference.

The Fallback: If the AI hits a rare snag, the node is hardcoded to fallback to a high-quality description.

📦 Installation
Place the SeansOmniTag folder into your ComfyUI/custom_nodes/ directory.

The node will automatically attempt to install required dependencies (transformers, whisper, opencv, etc.) on the first launch.

Restart ComfyUI and find the node under Sean's OmniTag 🛠️

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/seanhan19911990-source/ComfyUI-Seans-OmniTag.git
