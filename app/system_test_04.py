import soundfile
import nltk
import torch
import time

from sympy import false

print("torch version: " + torch.__version__)

from espnet2.bin.tts_inference import Text2Speech
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print("torch_xla version: " + torch_xla.__version__)
except:
    xm = None

# one more requirement
nltk.download('averaged_perceptron_tagger_eng')

device = "xla" if xm else "cuda"
# device = "cuda"  # force cuda use for comparison if desired

print(f"TTS generation using: {device.upper()}")

print("01: before model load")
# Prepare the model
text2speech = Text2Speech.from_pretrained("kan-bayashi/ljspeech_vits", device=device)

# now we can begin the performance test

def run_test_phrases(tts_model, phrases, cycle=0):
    total_elapsed_time = 0.0
    min_latency = 99.0
    min_offset = 0
    max_latency = 0.0
    max_offset = 0
    count = 1
    checking_output = True
    for phrase in phrases:
        start_time = time.perf_counter()
        # Run the model
        speech_output = tts_model(phrase)["wav"]
        # xm.mark_step()
        end_time = time.perf_counter()

        if count == 1:
            # Save the output audio
            soundfile.write(f"./output/test-set-{cycle}.{phrase}.{device}.wav", speech_output.detach().cpu().numpy(), text2speech.fs, "PCM_16")

        # capture metrics

        elapsed_time = end_time - start_time
        total_elapsed_time += elapsed_time
        if elapsed_time < min_latency:
            min_latency = elapsed_time
            min_offset = count - 1
        if elapsed_time > max_latency:
            max_latency = elapsed_time
            max_offset = count - 1
        print(f"completed: {count}/{len(phrases)}:\t phrase latency: {elapsed_time:0.3}")
        count = count + 1
    print(f"total elapsed time: {total_elapsed_time:0.3}")
    print(f"mean elapsed time: {(total_elapsed_time / len(phrases)):0.3}")
    print(f"minimum elapsed time: {min_latency:0.3}:\t phase: {phrases[min_offset]}")
    print(f"maximum elapsed time: {max_latency:0.3}\t phase: {phrases[max_offset]}")


test_phrases = [
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
    "I just want to say hello world",
]



print("\nStarting test one:")
print("------------------:")
run_test_phrases(text2speech, test_phrases, 0)

print("\nStarting test one (second time):")
print("------------------:")
run_test_phrases(text2speech, test_phrases, 1)

print("\nStarting test one (third time):")
print("------------------:")
run_test_phrases(text2speech, test_phrases, 2)

print("\nStarting test one (third time):")
print("------------------:")
run_test_phrases(text2speech, test_phrases, 3)
