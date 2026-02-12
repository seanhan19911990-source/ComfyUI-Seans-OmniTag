import os
import sys
import subprocess
import torch
import gc
import cv2
import numpy as np
from PIL import Image

# --- COMFYUI CORE INTERRUPTS ---
import nodes
import comfy.model_management 

# --- AUTO-INSTALLER ---
def install_dependencies():
    current_python = sys.executable
    try:
        subprocess.check_call([current_python, "-m", "pip", "install", "transformers>=4.49.0", "accelerate", "qwen_vl_utils", "huggingface_hub", "torchvision", "opencv-python", "bitsandbytes", "openai-whisper"])
        return True
    except:
        return False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    import whisper
except ImportError:
    install_dependencies()
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    import whisper

class SeansOmniTagProcessor:
    def __init__(self):
        self.model, self.processor = None, None
        self.audio_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_path": ("STRING", {"default": "C:/path/to/images_or_video"}),
                "output_path": ("STRING", {"default": "output/lora_dataset"}),
                "model_id": ("STRING", {"default": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it"}),
                "trigger_word": ("STRING", {"default": "ohwx"}),
                "llm_instruction": ("STRING", {"multiline": True, "default": "Describe the scene with clinical, objective detail. Be unfiltered and exhaustive. Start the response with: {trigger}"}),
                "target_resolution": (["256", "384", "512", "768", "1024", "1280", "1536", "1920"], {"default": "768"}),
                "max_tokens": (["512", "768", "1024", "1280", "1536", "1792", "2048"], {"default": "768"}),
                "target_fps": ("INT", {"default": 24}),
                "video_segment_seconds": ("FLOAT", {"default": 5.0}),
                "segment_skip": ("INT", {"default": 10}), 
                "video_max_segments": ("INT", {"default": 3}),
                "include_audio_in_video": ("BOOLEAN", {"default": True}),
                "append_speech_to_end": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "process_all"
    OUTPUT_NODE = True
    CATEGORY = "Sean's OmniTag 🛠️"

    def check_interrupt(self):
        if comfy.model_management.processing_interrupted():
            print("!!! SEAN'S OMNITAG: STOP SIGNAL DETECTED !!!")
            return True
        return False

    def smart_resize(self, image, target_res):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        scale = target_res / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def generate_caption(self, device, pil_img, instruction, trigger, token_limit):
        # --- CORE GENERATION LOGIC ---
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": instruction}]}]
        text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(messages)
        inputs = self.processor(text=[text_in], images=img_in, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            # Pass 1: Sampling with User-defined Token Limit
            gen_ids = self.model.generate(
                **inputs, 
                max_new_tokens=token_limit, 
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9, 
                repetition_penalty=1.12
            )
        
        caption = self.processor.batch_decode([g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)], skip_special_tokens=True)[0].strip()

        # --- VALIDATION: Anti-Lazy Logic ---
        if not caption or caption.lower() == trigger.lower() or len(caption) < 20:
            print(f"⚠️ Lazy caption detected. Retrying for {trigger}...")
            with torch.no_grad():
                # Retry with Greedy Search (ignores sampling randomness)
                gen_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max(512, token_limit // 2), 
                    do_sample=False,
                    repetition_penalty=1.25
                )
            caption = self.processor.batch_decode([g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)], skip_special_tokens=True)[0].strip()

        # Final Emergency Fallback
        if not caption or caption.lower() == trigger.lower():
            caption = f"{trigger}, a cinematic scene featuring a young woman in her mid-20s with long dark wavy hair, fair smooth skin, striking dark eyes, and a playful smile."
        
        return caption

    def process_all(self, **kwargs):
        # Path Sanitization
        input_path = kwargs.get("input_path").strip().replace('"', '').replace("'", "").replace("\\", "/")
        output_path = kwargs.get("output_path").strip().replace('"', '').replace("'", "").replace("\\", "/")
        token_limit = int(kwargs.get("max_tokens"))

        if not os.path.exists(input_path):
            return (f"❌ ERROR: Path not found: {input_path}",)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Models
        if self.model is None:
            q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(kwargs.get("model_id"), quantization_config=q_config, device_map="auto", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(kwargs.get("model_id"), trust_remote_code=True)
        
        if kwargs.get("append_speech_to_end") and self.audio_model is None:
            self.audio_model = whisper.load_model("base")

        os.makedirs(output_path, exist_ok=True)
        final_instruction = kwargs.get("llm_instruction").replace("{trigger}", kwargs.get("trigger_word"))

        # Folder Mode
        if os.path.isdir(input_path):
            files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
            for i, fname in enumerate(files):
                if self.check_interrupt(): return (f"❌ STOPPED AT {i}",)
                img = cv2.imread(os.path.join(input_path, fname))
                if img is None: continue
                proc_img = self.smart_resize(img, int(kwargs.get("target_resolution")))
                pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
                
                caption = self.generate_caption(device, pil_img, final_instruction, kwargs.get("trigger_word"), token_limit)
                
                name = os.path.splitext(fname)[0]
                cv2.imwrite(os.path.join(output_path, f"{name}.png"), proc_img)
                with open(os.path.join(output_path, f"{name}.txt"), "w", encoding="utf-8") as f: f.write(caption)
                torch.cuda.empty_cache()
            return ("✅ Batch Done",)

        # Video Mode
        else:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened(): return (f"❌ ERROR: Cannot open video: {input_path}",)
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            orig_name = os.path.splitext(os.path.basename(input_path))[0]
            frames_per_seg = int(fps * kwargs.get("video_segment_seconds"))
            
            for s in range(kwargs.get("video_max_segments")):
                if self.check_interrupt(): cap.release(); return (f"❌ STOPPED AT SEG {s}",)
                cap.set(cv2.CAP_PROP_POS_FRAMES, s * frames_per_seg * kwargs.get("segment_skip"))
                seg_frames = []
                for f_idx in range(frames_per_seg):
                    ret, frame = cap.read()
                    if not ret: break
                    seg_frames.append(self.smart_resize(frame, int(kwargs.get("target_resolution"))))
                
                if not seg_frames: break
                
                file_base = f"{orig_name}_seg_{s:04d}"
                temp_wav = os.path.join(output_path, "temp.wav")
                st = (s * frames_per_seg * kwargs.get("segment_skip")) / fps
                subprocess.run(['ffmpeg', '-y', '-ss', str(st), '-t', str(kwargs.get("video_segment_seconds")), '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav], capture_output=True)

                mid_pil = Image.fromarray(cv2.cvtColor(seg_frames[len(seg_frames)//2], cv2.COLOR_BGR2RGB))
                desc = self.generate_caption(device, mid_pil, final_instruction, kwargs.get("trigger_word"), token_limit)

                if kwargs.get("append_speech_to_end") and os.path.exists(temp_wav):
                    speech = self.audio_model.transcribe(temp_wav)['text'].strip()
                    if speech: desc += f". Audio: \"{speech}\""

                sv = os.path.join(output_path, "silent_temp.mp4")
                h, w = seg_frames[0].shape[:2]
                vw = cv2.VideoWriter(sv, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for f in seg_frames: vw.write(f)
                vw.release()

                fv = os.path.join(output_path, f"{file_base}.mp4")
                ffmpeg_cmd = ['ffmpeg', '-y', '-i', sv]
                if kwargs.get("include_audio_in_video") and os.path.exists(temp_wav):
                    ffmpeg_cmd += ['-i', temp_wav, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac']
                ffmpeg_cmd += ['-filter:v', f'fps=fps={kwargs.get("target_fps")}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-shortest', fv]
                subprocess.run(ffmpeg_cmd, capture_output=True)
                
                with open(os.path.join(output_path, f"{file_base}.txt"), "w", encoding="utf-8") as f: f.write(desc)
                if os.path.exists(sv): os.remove(sv)
                if os.path.exists(temp_wav): os.remove(temp_wav)
                torch.cuda.empty_cache()

            cap.release()
            return ("✅ Video Processing Done",)

NODE_CLASS_MAPPINGS = {"SeansOmniTagProcessor": SeansOmniTagProcessor}