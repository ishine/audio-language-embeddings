import os
import torch
from transformers import AutoModel, AutoTokenizer, set_seed
from ale.music2lang.modules.audio_rep import TFRep
from ale.music2lang.modules.tokenizer import ResFrontEnd, SpecPatchEmbed
from ale.music2lang.modules.encoder import MusicTransformer
from ale.music2lang.modules.model import ContrastiveModel
from ale.constants import ASSET, MUSIC2LANG_FILENAME

def get_model():
    audio_preprocessr = TFRep(
                sample_rate= 16000,
                f_min=0,
                f_max= int(16000 / 2),
                n_fft = 1024,
                win_length = 1024,
                hop_length = int(0.01 * 16000),
                n_mels = 128
    )

    frontend = ResFrontEnd(
        input_size=(128, int(100 * 9.91) + 1), # 128 * 992
        conv_ndim=128, 
        attention_ndim=256,
        mix_type= "cf"
    )
    audio_encoder = MusicTransformer(
        audio_representation=audio_preprocessr,
        frontend = frontend,
        audio_rep = "mel",
        attention_nlayers= 4,
        attention_ndim= 256
    )
    text_encoder = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = ContrastiveModel(
        audio_encoder= audio_encoder,
        text_encoder= text_encoder,
        text_type = "stocastic",
        audio_dim= 256,
        text_dim= 768,
        mlp_dim= 128,
        temperature = 0.2
    )
    pretrained_object = torch.load(os.path.join(ASSET, MUSIC2LANG_FILENAME), map_location='cpu')
    state_dict = pretrained_object['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer