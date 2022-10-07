import os
from os import path
from vosk import Model, KaldiRecognizer
import json
import shlex
import subprocess
from multiprocessing import Pool
from pqdm.processes import pqdm

FISHER_DIR = "/home/mmcneil/datasets/fisher_corpus/"
JSON_DIR = path.join(FISHER_DIR, "vosk")

model_path = "vosk-model-en-us-0.22"
model = Model(model_path, lang='en-us')

SAMPLE_RATE = 16000


def resample_ffmpeg(infile, channel):
    cmd_str = (
        "ffmpeg -nostdin -loglevel quiet  "
        + "-i '{}' -filter_complex \"[0:a]channelsplit=channel_layout=stereo:channels=F"
        + channel
        + '[right]" -map "[right]"  -ac 1 -ar {} -f s16le -'
    ).format(str(infile), SAMPLE_RATE)
    cmd = shlex.split(cmd_str)
    stream = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return stream


WAV_DIR = path.join(FISHER_DIR, "wav")
F = 4000


def process(wavfile):
    wav_id = wavfile.replace(".wav", "")
    json_left = path.join(JSON_DIR, f"{wav_id}-left.json")
    json_right = path.join(JSON_DIR, f"{wav_id}-right.json")

    wavpath = path.join(WAV_DIR, wavfile)

    stream_left = resample_ffmpeg(wavpath, "L").stdout
    rec_left = KaldiRecognizer(model, 16000.0)
    rec_left.SetWords(True)

    stream_right = resample_ffmpeg(wavpath, "R").stdout
    rec_right = KaldiRecognizer(model, 16000.0)
    rec_right.SetWords(True)

    streaming_l = True
    streaming_r = True

    results_l = []
    results_r = []

    while streaming_l and streaming_r:
        if streaming_l:
            left_chunk = stream_left.read(4000)
            if len(left_chunk) > 0:
                if rec_left.AcceptWaveform(left_chunk):
                    res = json.loads(rec_left.Result())
                    if res["text"]:
                        results_l.append(res)
            else:
                res = json.loads(rec_left.FinalResult())
                if res['text']:
                    results_l.append(res)
                streaming_l = False


        if streaming_r:
            right_chunk = stream_right.read(4000)
            if len(right_chunk) > 0:
                if rec_right.AcceptWaveform(right_chunk):
                    res = json.loads(rec_right.Result())
                    if res["text"]:
                        results_r.append(res)
            else:
                res = json.loads(rec_right.FinalResult())
                if res['text']:
                    results_r.append(res)
                streaming_r = False


    with open(json_left, "w") as outfile:
        json.dump(results_l, outfile)
    with open(json_right, "w") as outfile:
        json.dump(results_r, outfile)

    return results_l


if __name__ == "__main__":
    try:
        os.mkdir(JSON_DIR)
    except:
        pass

    json_files = set(
        [
            x.replace("-left.json", "").replace("-right.json", "")
            for x in os.listdir(JSON_DIR)
        ]
    )

    wav_files = [
        x
        for x in os.listdir(WAV_DIR)
        if ".wav" in x and x.replace(".wav", "") not in json_files
    ]

    pqdm(wav_files, process, n_jobs=16)
