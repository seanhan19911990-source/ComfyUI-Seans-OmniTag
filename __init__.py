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

    def process_all(self, **kwargs):
        # --- PATH SANITIZATION ---
        input_path = kwargs.get("input_path").strip().replace('"', '').replace("'", "")
        input_path = os.path.normpath(input_path).replace("\\", "/")
        
        output_path = kwargs.get("output_path").strip().replace('"', '').replace("'", "")
        output_path = os.path.normpath(output_path).replace("\\", "/")

        if not os.path.exists(input_path):
            return (f"❌ ERROR: Path not found! Check spelling: {input_path}",)

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

        # --- IMAGE FOLDER LOGIC ---
        if os.path.isdir(input_path):
            files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
            if not files: return ("❌ ERROR: No images found in folder.",)
            
            for i, fname in enumerate(files):
                if self.check_interrupt(): return ("❌ STOPPED AT FILE " + str(i),)
                raw_img = cv2.imread(os.path.join(input_path, fname))
                if raw_img is None: continue
                proc_img = self.smart_resize(raw_img, int(kwargs.get("target_resolution")))
                pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
                messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": final_instruction}]}]
                text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, _ = process_vision_info(messages)
                inputs = self.processor(text=[text_in], images=img_in, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_ids = self.model.generate(**inputs, max_new_tokens=512)
                caption = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                name = os.path.splitext(fname)[0]
                cv2.imwrite(os.path.join(output_path, f"{name}.png"), proc_img)
                with open(os.path.join(output_path, f"{name}.txt"), "w", encoding="utf-8") as f: f.write(caption)
                torch.cuda.empty_cache()
            return ("✅ Batch Done",)

        # --- VIDEO FILE LOGIC ---
        else:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return (f"❌ ERROR: OpenCV cannot open video. Check path/format: {input_path}",)
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            orig_name = os.path.splitext(os.path.basename(input_path))[0]
            frames_per_seg = int(fps * kwargs.get("video_segment_seconds"))
            
            for s in range(kwargs.get("video_max_segments")):
                if self.check_interrupt(): cap.release(); return ("❌ STOPPED AT SEGMENT " + str(s),)
                cap.set(cv2.CAP_PROP_POS_FRAMES, s * frames_per_seg * kwargs.get("segment_skip"))
                seg_frames = []
                for f_idx in range(frames_per_seg):
                    if f_idx % 5 == 0 and self.check_interrupt(): cap.release(); return ("❌ STOPPED MID-GRAB",)
                    ret, frame = cap.read()
                    if not ret: break
                    seg_frames.append(self.smart_resize(frame, int(kwargs.get("target_resolution"))))
                
                if not seg_frames: break
                
                file_base = f"{orig_name}_seg_{s:04d}"
                temp_wav = os.path.join(output_path, "temp.wav")
                
                # Extract Audio with FFmpeg
                if kwargs.get("append_speech_to_end") or kwargs.get("include_audio_in_video"):
                    st = (s * frames_per_seg * kwargs.get("segment_skip")) / fps
                    subprocess.run(['ffmpeg', '-y', '-ss', str(st), '-t', str(kwargs.get("video_segment_seconds")), '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav], capture_output=True)

                # Captioning
                mid_pil = Image.fromarray(cv2.cvtColor(seg_frames[len(seg_frames)//2], cv2.COLOR_BGR2RGB))
                messages = [{"role": "user", "content": [{"type": "image", "image": mid_pil}, {"type": "text", "text": final_instruction}]}]
                text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, _ = process_vision_info(messages)
                inputs = self.processor(text=[text_in], images=img_in, padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_ids = self.model.generate(**inputs, max_new_tokens=512)
                desc = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

                if kwargs.get("append_speech_to_end") and os.path.exists(temp_wav):
                    speech = self.audio_model.transcribe(temp_wav)['text'].strip()
                    if speech: desc += f". Audio: \"{speech}\""

                # Final Video Assembly
                sv = os.path.join(output_path, "silent_temp.mp4")
                h, w = seg_frames[0].shape[:2]
                vw = cv2.VideoWriter(sv, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for f in seg_frames: vw.write(f)
                vw.release()

                fv = os.path.join(output_path, f"{file_base}.mp4")
                if kwargs.get("include_audio_in_video") and os.path.exists(temp_wav):
                    subprocess.run(['ffmpeg', '-y', '-i', sv, '-i', temp_wav, '-filter:v', f'fps=fps={kwargs.get("target_fps")}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-shortest', fv], capture_output=True)
                else:
                    subprocess.run(['ffmpeg', '-y', '-i', sv, '-filter:v', f'fps=fps={kwargs.get("target_fps")}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', fv], capture_output=True)
                
                with open(os.path.join(output_path, f"{file_base}.txt"), "w", encoding="utf-8") as f: f.write(desc)
                if os.path.exists(sv): os.remove(sv)
                if os.path.exists(temp_wav): os.remove(temp_wav)
                torch.cuda.empty_cache()

            cap.release()
            return ("✅ Video Done",)

NODE_CLASS_MAPPINGS = {"SeansOmniTagProcessor": SeansOmniTagProcessor}