# Audio-Langauge Embeddings

This project maps `language` and `audio` to the joint embedding space. A wrapping repository for several public models.


## Available Models

|            |   Datset  |  Size (Hour) |    SR   | Vocab | Modality | Support |
|:----------:|:---------:|:------------:|:-------:|:-----:|:--------:|:-------:|
|  AudioCLIP |  [Audioset](https://research.google.com/audioset/) | 1.8M (5000h) | 44.1kHz |  527  |   A,V,T  |    ✅    |
|  wav2CLIP  |  [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) |  200k (556h) |  16kHz  |  309  |   A,V,T  |         |
| Music2Lang | [MSD-eCALS](https://github.com/SeungHeonDoh/msd-splits) | 517k (4300h) |  16kHz  |  1054 |    A,T   |    ✅    |

- WAV2CLIP: LEARNING ROBUST AUDIO REPRESENTATIONS FROM CLIP: https://arxiv.org/pdf/2110.11499
- AudioCLIP: Extending CLIP to Image, Text and Audio: https://arxiv.org/abs/2106.13043
- Music2Langauge: Ours

> ###### Issue:
> 
> <sub>• WAV2CLIP: currently only open audio embedding weights.<br /> • CLAP: There are no published codes and weights yet..<br /></sub>


## Quickstart
Requirements: >1 GPU
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


## Reference

- AudioCLIP: https://github.com/AndreyGuzhov/AudioCLIP
- wav2CLIP: https://github.com/descriptinc/lyrebird-wav2clip
- CLAP: https://arxiv.org/abs/2206.04769 (not open)