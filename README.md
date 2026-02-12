# ComfyUI-Seans-OmniTag

# 🛠️ Sean's OmniTag: The Ultimate LTX-2 Dataset Tool

**Sean's OmniTag** is a powerhouse ComfyUI node designed specifically for creators building datasets for **LTX-Video (LTX-2)**, **Flux**, and high-fidelity video LoRAs. It automates the most painful parts of data prep: resampling, resizing, visual captioning, and audio transcription.

## 🚀 Why use this?


* **🎬 LTX-2 Standardized:** Automatically resamples video segments to **24 FPS**, ensuring your training data matches the LTX-Video motion model perfectly.
* **💎 True HD Ladder:** Support for resolutions from **256px** all the way to **1920px (1080p)**. It uses smart-aspect scaling to maintain quality without distorting your subjects.
* **🧠 Multimodal Intelligence:** * **Visuals:** Powered by `Qwen2.5-VL` for hyper-detailed, clinical descriptions.
    * **Audio:** Powered by `OpenAI Whisper` to transcribe dialogue directly into your tags.
 
    * 
* **📂 Batch Workflow:** Point it at a folder of images or a single long-form video, and it will churn out paired `.png/.mp4` and `.txt` files ready for training.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
cd ComfyUI/custom_nodes
git clone [https://github.com/YOUR_USERNAME/ComfyUI-Seans-OmniTag](https://github.com/YOUR_USERNAME/ComfyUI-Seans-OmniTag)
