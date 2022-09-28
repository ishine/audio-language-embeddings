ASSET = "../dataset/pretrained"
OUTPUT_PATH = "../dataset/output"

AudioCLIP_FILENAME = 'AudioCLIP-Full-Training.pt'
AudioCLIP_BPE = 'bpe_simple_vocab_16e6.txt.gz'
AudioCLIP_SAMPLE_RATE = 44100
AudioCLIP_DURATION = 10
IMAGE_SIZE = 224
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

WAV2CLIP_SAMPLING_RATE = 16000

MUSIC2LANG_FILENAME = "music2lang.pth"
MUSIC2LANG_SAMPLING_RATE = 16000
MUSIC2LANG_DURATION = 9.91

SAMPLE_A = ["../dataset/samples/060242.mp3", "../dataset/samples/096838.mp3"]
SAMPLE_T = "../dataset/samples/text.json"