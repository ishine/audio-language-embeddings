# Audio-Language Embedding Extractor

This project maps `language` and `audio` to the joint embedding space. A wrapping repository for several public models.


<p align = "center">
    <img src = "https://i.imgur.com/PLLAXNt.png">
</p>

## Available Models

|  | Datset | Domain | Size | SR | Vocab | Modality | Mapping | Available |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AudioCLIP |<sub>Audioset</sub> | <sub>Env,Speech,Music</sub> | 1.8M (5000h) | 44.1kHz | 527 | A,V,T | A-V, A-T | ✅ |
| Wav2CLIP | <sub>VGGSound</sub> | <sub>Env,Speech,Music</sub>  | 200k (556h) | 16kHz | 309 | A,V,T | A-V |  |
| CLAP | <sub>FSD50k, ClothoV2</sub> <br> <sub>AudioCaps, MACS</sub> | <sub>Env,Speech,Music</sub>  | 128k (-) | 44.1KHz | - | A,T | A-T |  |
| Ours | <sub>MSD-eCALS</sub> | <sub>Music</sub>  | 517k (4300h) | 16kHz | 1054 | A,T | A-T | ✅ |
| MusCALL | <sub>Private (Universal Music)</sub> | <sub>Music</sub>  | 250k (-) | 16kHz | - | A,T | A-T |  |
| Mulan | <sub>Private (Goolge)</sub> | <sub>Music</sub>  | 44M (370000h) | - | - | A,T | A-T |  |


- AudioCLIP: Extending CLIP to Image, Text and Audio: https://arxiv.org/abs/2106.13043
- WAV2CLIP: LEARNING ROBUST AUDIO REPRESENTATIONS FROM CLIP: https://arxiv.org/pdf/2110.11499
- CLAP : LEARNING AUDIO CONCEPTS FROM NATURAL LANGUAGE SUPERVISION: https://arxiv.org/pdf/2206.04769
- Mus2Lang: Ours
- MusCALL : Contrastive Audio-Language Learning for Music: https://arxiv.org/abs/2208.12208
- MuLan: MuLan: A Joint Embedding of Music Audio and Natural Language: https://arxiv.org/abs/2208.12415

> ###### Issue:
> 
> <sub>• WAV2CLIP: currently only open audio embedding weights.<br /> • CLAP,MusCALL,MuLan: There are no published codes and weights yet..<br /></sub>


## Quickstart
Download your pytorch 1.8.1 at https://pytorch.org/get-started/previous-versions/

```
conda install -c conda-forge pysoundfile    # pysoundfile-0.10.3 important for liborsa!
conda install -c conda-forge ffmpeg         # ffmpeg-4.3.2 is important for liborsa!

pip3 install -e .
cd scripts
bash download_pretrain.sh
```

## Embedding Inference
We put some audio files and examples of text pairs in `dataset/samples`. 
The `extractor.py` function extracts the audio-language joint embedding `z_audio, z_text`.

```
cd ale
python extractor.py --inference_type audio --model_type audioclip
python extractor.py --inference_type audio --model_type music2lang
python extractor.py --inference_type text --model_type audioclip
python extractor.py --inference_type text --model_type music2lang
```

## Check Params
See `ale/constants`

```
ASSET = "../dataset/pretrained"
OUTPUT_PATH = "../dataset/output"

AudioCLIP_FILENAME = 'AudioCLIP-Full-Training.pt'
AudioCLIP_BPE = 'bpe_simple_vocab_16e6.txt.gz'
AudioCLIP_SAMPLE_RATE = 44100
AudioCLIP_DURATION = 10

WAV2CLIP_SAMPLING_RATE = 16000

MUSIC2LANG_FILENAME = "music2lang.pth"
MUSIC2LANG_SAMPLING_RATE = 16000
MUSIC2LANG_DURATION = 9.91
SAMPLE_A = ["../dataset/samples/060242.mp3", "../dataset/samples/096838.mp3"]
SAMPLE_T = "../dataset/samples/text.json"
```


## Code Reference

- AudioCLIP: https://github.com/AndreyGuzhov/AudioCLIP
- wav2CLIP: https://github.com/descriptinc/lyrebird-wav2clip
