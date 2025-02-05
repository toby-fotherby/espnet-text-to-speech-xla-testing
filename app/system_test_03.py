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

# one more requirement - needs to run once after the installs
nltk.download('averaged_perceptron_tagger_eng')

device = "xla" if xm else "cuda"
# device = "cuda"  # force cuda use for comparison if desired

print(f"TTS generation using: {device.upper()}")

print("01: before model load")
# Prepare the model
text2speech = Text2Speech.from_pretrained("kan-bayashi/ljspeech_vits", device=device)

print("02: before first model run")

text = "version two four compiled: you are happy"

# Run the model
speech = text2speech(text)["wav"]

# exit()

print("03: before write output")
xm.mark_step()

# Save the output audio
soundfile.write(f"./output/{text}.{device}.wav", speech.detach().cpu().numpy(), text2speech.fs, "PCM_16")

# now we can begin the performance test

def run_test_phrases(tts_model, phrases):
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
        xm.mark_step()
        end_time = time.perf_counter()

        if True:
            # make sure the output generated is valid
            # xm.mark_step() - moved to line above

            # Save the output audio
            soundfile.write(f"./output/{phrase}.{device}.wav", speech_output.detach().cpu().numpy(), text2speech.fs, "PCM_16")

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
    "I am bold",
    "I am cold",
    "I am fold",
    "I am gold",
    "I am mold",
    "I am sold",
    "I am told",
]

more_test_phrases = [
    "The girl drew a cat.",
    "That paint is still wet.",
    "The colors blend well.",
    "The photo is so sharp.",
    "Her voice is beautiful.",
    "The song moved me to tears.",
    "He plays piano well.",
    "The dance was graceful.",
    "The statue looked lifelike.",
    "The light enhanced the art.",

    "The brush strokes were thick.",
    "The canvas needs priming first.",
    "The texture adds interest.",
    "The drama held my interest.",
    "The scene portrayed sadness.",
    "The melody was cheerful.",
    "The tempo change surprised me.",
    "The lyrics were thoughtful.",
    "The costume design impressed me.",
    "The choreography told a story.",

    "The acting was convincing.",
    "The set design was elaborate.",
    "The director had a vision.",
    "The composition was balanced.",
    "The tone poem was evocative.",
    "The portrait captured her essence.",
    "The motif recurred throughout.",
    "The canvas came alive with vibrant hues.",
    "The imagery was vivid.",
    "The exhibition was fascinating.",

    "art moves the soul with life,",
    "`Paint strokes call out to me,",
    "Clay's forms reveal beauty's truth,",
    "Sculptures spark imaginations,",
    "Music paints with melodies,",
    "Brushes dance across canvas,",
    "Words weave tapestries of thought,",
    "Cameras capture moments framed,",
    "Stage lights illuminate passion,",

    "Pencils sketch dreams onto page,",
    "Art expresses what words can't,",
    "Colors blend in harmonies,",
    "Rhythms pulse with creative fire,",
    "Beauty blooms from artist's hand,",
    "Masterpieces inspire awe,",
    "Art transcends boundaries,",
    "Creativity knows no bounds,",
    "Galleries house vision's light,",
    "Sculpture chisels stories cold,",

    "Murals paint walls with life,",
    "Art speaks in universal tongues,",
    "Emotions flow from palette's wells,",
    "Performances captivate hearts,",
    "Crafters weave magic threads,",
    "Poetry paints with words' hues,",
    "Art celebrates life's canvas,",
    "Beauty resonates deep chords,",
    "Creatives birth new worlds,",
    "Art nurtures the human soul,"
]

print("\nStarting test one:")
print("------------------:")
run_test_phrases(text2speech, test_phrases)
# exit(1)

print("\nStarting test one (again):")
print("------------------:")
run_test_phrases(text2speech, test_phrases)

exit(0)

print("\nStarting test two:")
print("------------------:")
run_test_phrases(text2speech, more_test_phrases)

print("\nStarting test two (again:")
print("------------------:")
run_test_phrases(text2speech, more_test_phrases)

print("\nStarting test one (again):")
print("------------------:")
run_test_phrases(text2speech, test_phrases)