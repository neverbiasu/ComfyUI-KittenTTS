from typing import Tuple, Dict
import os
from comfy.comfy_types import IO
from .get_model import KittenTTS
import torch


class KittenTTSLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Zero-config loader: no inputs required
        return {"required": {}}

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "load"
    CATEGORY = "audio/loaders"

    def load(self):
        # Use defaults defined in get_model.KittenTTS
        tts = KittenTTS()
        return (tts,)  # IO.MODEL carrying the Python object
  

class KittenTTSSampler:
    VOICES = [
        "expr-voice-2-m", "expr-voice-2-f", "expr-voice-3-m", "expr-voice-3-f",
        "expr-voice-4-m", "expr-voice-4-f", "expr-voice-5-m", "expr-voice-5-f",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_model": (IO.MODEL, {"forceInput": True}),
                "text": (IO.STRING, {"multiline": True, "default": ""}),
                "voice": (IO.COMBO, {"default": "expr-voice-5-m", "options": cls.VOICES}),
                "speed": (IO.FLOAT, {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = (IO.AUDIO,)
    FUNCTION = "generate"
    CATEGORY = "audio/tts"

    def generate(self, tts_model, text: str, voice: str, speed: float) -> Tuple[Dict]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        wav_np = tts_model.generate(text, voice=voice, speed=speed)
        if wav_np.ndim != 1:
            wav_np = wav_np.reshape(-1)
        waveform = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0)  # [B=1, C=1, T]
        audio = {"waveform": waveform, "sample_rate": 24000}
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "KittenTTSLoader": KittenTTSLoader,
    "KittenTTSSampler": KittenTTSSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KittenTTSLoader": "KittenTTS Loader",
    "KittenTTSSampler": "KittenTTS Sampler",
}