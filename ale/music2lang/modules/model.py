import torch
import torchaudio
import math
import numpy as np
from torch import Tensor, nn

class ContrastiveModel(nn.Module):
    def __init__(self, audio_encoder, text_encoder, text_type, audio_dim, text_dim, mlp_dim, temperature):
        super(ContrastiveModel, self).__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.text_type = text_type
        self.init_temperature = torch.tensor([np.log(1/temperature)])
        self.logit_scale = nn.Parameter(self.init_temperature, requires_grad=True)
        self.head = CLIPHead(logit_scale=self.logit_scale)
        self.audio_projector = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, mlp_dim, bias=False))
        self.text_projector =  nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, mlp_dim, bias=False))
        self.audio_encoder.train()
        self.text_encoder.train()
        self.a_latent = nn.Identity()
        self.t_latent = nn.Identity()

    def forward(self, audio, text, text_mask=None):
        h_audio = self.encode_audio(audio)
        if self.text_type == "bert":
            h_text = self.encode_bert_text(text, text_mask)
        elif self.text_type == "glove":
            h_text = self.encode_glove_text(text)
        audio_loss = self.head(h_audio, h_text)
        text_loss = self.head(h_text, h_audio)
        loss = (audio_loss + text_loss) / 2

        audio_acc = self.head.acc(h_audio, h_text)
        text_acc = self.head.acc(h_text, h_audio)
        return loss, audio_acc, text_acc, self.logit_scale
        
    def encode_audio(self, audio):
        # audio = (Batch x Length x Dim)
        audio_emb = self.audio_encoder(audio)
        h_audio = self.a_latent(audio_emb[:,0,:])
        z_audio = self.audio_projector(h_audio)
        return z_audio

    def encode_bert_text(self, text, text_mask):
        text_emb = self.text_encoder(input_ids=text, attention_mask=text_mask)
        h_text = self.t_latent(text_emb['last_hidden_state'][:,0,:])
        z_text = self.text_projector(h_text)
        return z_text

    def encode_glove_text(self, text_emb): 
        h_text = self.t_latent(text_emb)
        z_text = self.text_projector(h_text)
        return z_text
    

class CLIPHead(nn.Module):
    def __init__(self, logit_scale):
        super(CLIPHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.logit_scale = logit_scale

    def forward(self, h1, h2):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            h1 = SyncFunction.apply(h1)
            h2 = SyncFunction.apply(h2)
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm', [h1, h2]) * temperature.to(device)
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)
    
    def acc(self, h1, h2):
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm', [h1, h2]) * temperature.to(device)
        N = logits.shape[0]  # batch size per GPU
        y_pred = logits.max(dim=-1)[1]
        target = torch.arange(N, dtype=torch.long, device=device)
        train_acc = torch.sum(y_pred == target)
        acc = train_acc / N
        return acc