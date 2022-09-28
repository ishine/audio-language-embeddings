import os
import json
import torch
import librosa
import numpy as np
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from ale.audioclip.utils.transforms import ToTensor1D
from ale.audioclip.model import AudioCLIP
from ale import wav2clip
from ale import music2lang
from ale.constants import ASSET, AudioCLIP_FILENAME, AudioCLIP_SAMPLE_RATE,AudioCLIP_DURATION, MUSIC2LANG_SAMPLING_RATE, MUSIC2LANG_DURATION, SAMPLE_A, SAMPLE_T, OUTPUT_PATH

def a2c_audio_transform(audio_path):
    audio, _ = librosa.load(audio_path, sr=AudioCLIP_SAMPLE_RATE, dtype=np.float32)
    input_size = int(AudioCLIP_DURATION * AudioCLIP_SAMPLE_RATE)
    hop = len(audio) // input_size
    audio = np.stack([np.array(audio[i * input_size : (i + 1) * input_size]) for i in range(hop)]).astype('float32')
    audio_input = torch.from_numpy(audio).unsqueeze(1) # batch, 1, lenght 
    return audio_input

def a2c_text_transform(text_input):
    return [[text] for text in text_input]

def a2c_audio_extractor(list_of_audio_path, a2c_model):
    """
    input: list_of_audio_path
    output: batch x embedding (b x 1024)
    """
    z_audios = []
    for audio_path in list_of_audio_path:
        audio_input = a2c_audio_transform(audio_path)
        ((z_audio, _, _), _), _ = a2c_model(audio=audio_input)
        z_audios.append(z_audio)
    return z_audios

def a2c_text_extractor(text_path, a2c_model):
    """
    input: list_of_text
    output: batch x embedding (b x 1024)
    """
    kv_data = json.load(open(text_path, 'r'))
    list_of_fname = list(kv_data.keys())
    list_of_text = [kv_data[fname] for fname in list_of_fname]
    text_input = a2c_text_transform(list_of_text)
    ((_, _, z_text), _), _ = a2c_model(text=text_input)
    return z_text


def m2l_audio_transform(audio_path):
    audio, _ = librosa.load(audio_path, sr=MUSIC2LANG_SAMPLING_RATE, dtype=np.float32)
    input_size = int(MUSIC2LANG_DURATION * MUSIC2LANG_SAMPLING_RATE)
    if len(audio) < input_size:
        audio_tensor = torch.from_numpy(audio).unqueeze(0)
    else:
        hop = len(audio) // input_size
        audio = np.stack([np.array(audio[i * input_size : (i + 1) * input_size]) for i in range(hop)]).astype('float32')
        audio_tensor = torch.from_numpy(audio)
    return audio_tensor

def m2l_text_transform(list_of_text, tokenzier):
    encoding = tokenzier.batch_encode_plus(list_of_text, padding='longest', max_length=64, truncation=True, return_tensors="pt")
    text = encoding['input_ids']
    text_mask = encoding['attention_mask']
    return text, text_mask

def m2l_audio_extractor(list_of_audio_path, m2l_model):
    """
    input: list_of_audio_path
    output: batch x embedding (b x 128)
    """
    z_audios = []
    for audio_path in list_of_audio_path:
        audio_input = m2l_audio_transform(audio_path)
        with torch.no_grad():
            z_audio = m2l_model.encode_audio(audio_input)
        z_audios.append(z_audio)
    return z_audios

def m2l_text_extractor(text_path, tokenzier, m2l_model):
    """
    input: text_path
    output: batch x embedding (b x 128)
    """
    kv_data = json.load(open(text_path, 'r'))
    list_of_fname = list(kv_data.keys())
    list_of_text = [kv_data[fname] for fname in list_of_fname]
    text, text_mask = m2l_text_transform(list_of_text, tokenzier)
    with torch.no_grad():
        z_text = m2l_model.encode_bert_text(text, text_mask)
    return list_of_text, z_text

def main(args):
    if args.model_type == "audioclip":
        a2c_model = AudioCLIP(pretrained=os.path.join(ASSET, AudioCLIP_FILENAME))
        if args.inference_type == "text":
            list_of_text, z_text = a2c_text_extractor(SAMPLE_T, a2c_model)
            output = {text:emb for text, emb in zip(list_of_text, z_text)}
        elif args.inference_type == "audio":
            z_audio = a2c_audio_extractor(SAMPLE_A, a2c_model)
            output = {text:emb for text, emb in zip(SAMPLE_A, z_audio)}
    elif args.model_type == "music2lang":
        m2l_model, m2l_tokenizer = music2lang.get_model()
        if args.inference_type == "text":
            list_of_text, z_text = m2l_text_extractor(SAMPLE_T, m2l_tokenizer, m2l_model)
            output = {text:emb for text, emb in zip(list_of_text, z_text)}
        elif args.inference_type == "audio":
            z_audio = m2l_audio_extractor(SAMPLE_A, m2l_model)
            output = {text:emb for text, emb in zip(SAMPLE_A, z_audio)}
    elif args.model_type == "wav2clip":
        w2c_model = wav2clip.get_model()
    print(f"total {len(output)} of {args.model_type}_{args.inference_type}_{args.tid} save in {OUTPUT_PATH} dir")
    torch.save(output, os.path.join(OUTPUT_PATH, f"{args.model_type}_{args.inference_type}_{args.tid}.pt"))

if __name__ == "__main__":
    # pipeline
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="audioclip", type=str)
    parser.add_argument("--inference_type", default="audio", type=str)
    parser.add_argument("--tid", default="sample", type=str)
    args = parser.parse_args()
    main(args)