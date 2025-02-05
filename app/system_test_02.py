import soundfile
import nltk

from espnet2.bin.tts_inference import Text2Speech
try:
    import torch_xla.core.xla_model as xm
except:
    xm = None

# one more requirement - needs to run once after the installs
nltk.download('averaged_perceptron_tagger_eng')

device = "xla" if xm else "cuda"
# device = "cuda"  # force cuda use for comparison if desired

print(f"TTS generation using: {device.upper()}")

print("01: before model load")
# Prepare the model
text2speech = Text2Speech.from_pretrained("kan-bayashi/ljspeech_vits", device=device)

print("02: before model run")

text = "version two four compiled: you are happy"

# Run the model
speech = text2speech(text)["wav"]

print("03: before write output")
xm.mark_step()

# Save the output audio
soundfile.write(f"./output/{text}.{device}.wav", speech.detach().cpu().numpy(), text2speech.fs, "PCM_16")