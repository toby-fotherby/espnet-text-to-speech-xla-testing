import soundfile
import nltk
from espnet2.bin.tts_inference import Text2Speech

# needed to patch import error with scipy.signal.windows in pqmf.py

# Test configuration with CUDA
device = "cuda"

print(f"TTS generation using: {device.upper()}")

# Prepare the model
text2speech = Text2Speech.from_pretrained("kan-bayashi/ljspeech_vits", device=device)
# one more requirement
nltk.download('averaged_perceptron_tagger_eng')

# Run the model
speech = text2speech("hello world")["wav"]

# Save the output audio
soundfile.write(f"{device}.out.wav", speech.detach().cpu().numpy(), text2speech.fs, "PCM_16")