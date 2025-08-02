import math

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from utils import first_element, generate_square_subsequent_mask,pair
import math



class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(torch.nn.Module):
    """Classic Attention-is-all-you-need positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.dropout = nn.Dropout(p=dropout)
        pe = self.make_pe(d_model, max_len)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(d_model: int, max_len: int) -> Tensor:
        """Compute positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[2] == self.pe.shape[2]  # type: ignore
        x = x + self.pe[: x.size(0)]  # type: ignore
        return self.dropout(x)





def dirichlet_sample(alpha):

    dist = torch.distributions.dirichlet.Dirichlet(alpha)
    return dist.sample()


class SpectrumAttention(nn.Module):

    def __init__(self, D, H, N, alpha=10.):
        super().__init__()
        self.pi = nn.Parameter(
            data=dirichlet_sample(
                alpha * torch.ones(N)),
            requires_grad=True)
        self.Q = nn.Parameter(
            data=torch.normal(
                0., 1., (D, H, N)), requires_grad=True)

    def forward(self, s):
        # spectrum is [B,D]
        # unbatched single head: Q is [D,H], QQT is [D,D]

        M = (self.Q @ self.pi.view(1, -1, 1)).squeeze(-1)
        MTs = M.transpose(0, 1).unsqueeze(0) @ s.unsqueeze(-1)
        MMTs = (M.unsqueeze(0) * MTs.permute(0,2,1))
        return MMTs


class CNN_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.model_cfg.d_model
        self.type_embedding = InputEmbeddings(config.model_cfg.d_model, config.vocab_size)
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.bottleneck = nn.Conv1d(256, self.d_model, 1)
        self.concatnet = nn.Conv1d(503, 500, 1)

    def forward(self, w,d):
        w = self.type_embedding(w) # [8,3,128]
        x = self.spectral_conv(d.unsqueeze(1).float())
        x = self.bottleneck(x)
        x = x.flatten(start_dim=2)
        x = x.permute(0, 2, 1)
        x = torch.cat((w, x), dim=1)
        x = self.concatnet(x)
        x = x.permute(1, 0, 2) #[500,8,128]
        return x

class Inverse_Encoder(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.d_model = config.model_cfg.d_model
        self.heads = config.model_cfg.heads
        self.emb_dropout =  config.model_cfg.emb_dropout
        self.vocab_size =  config.vocab_size
        self.dim_feedforward =  config.model_cfg.dim_feedforward
        self.encoder_dropout =  config.model_cfg.encoder_dropout
        self.num_encoder_layers =  config.model_cfg.num_encoder_layers

        self.Linear = nn.Linear(1, self.d_model)
        self.spectral_layer = SpectrumAttention(500,self.d_model,self.heads)
        self.param_embedding = nn.Linear(10, 500 * self.d_model)  # Embedding for parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, 503, self.d_model))

        self.dropout = nn.Dropout(self.emb_dropout)
        self.type_embedding = InputEmbeddings(self.d_model, self.vocab_size)
        

        self.word_positional_encoder = PositionalEncoding(self.d_model, max_len=2048)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.heads, dim_feedforward=self.dim_feedforward, dropout=self.encoder_dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, self.num_encoder_layers)
    def encode(self, d,w):
        d = self.Linear(d)
        cls_tokens = w
        x = torch.cat((cls_tokens, d), dim=1)

        x += self.pos_embedding

        x = self.dropout(x)
        x = self.transformer_encoder(x) #(8,504,124)

        return x

    def forward(self, w,d):
        
        w = self.type_embedding(w)
        x = self.encode(d.unsqueeze(-1),w)
        x = x.permute(1, 0, 2)

        # src_key_padding_mask = (w == 0).transpose(0, 1)
        # w = self.type_embedding(w)
        # w = self.word_positional_encoder(w)
        # x = self.transformer_encoder(w, src_key_padding_mask=src_key_padding_mask)
        # x = x.permute(1, 0, 2)

        return x  # (x,batch_size,d_model)




class Inerse_Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.max_output_len = config.trainer_cfg.max_output_len + 2
        self.d_model = config.model_cfg.d_model
        self.vocab_size = config.model_cfg.vocab_size
        self.heads = config.model_cfg.heads
        self.dim_feedforward = config.model_cfg.dim_feedforward
        self.decoder_dropout = config.model_cfg.decoder_dropout
        self.num_decoder_layers = config.model_cfg.num_decoder_layers
        self.tokenizer = config.tokenizer
        self.sos_index = config.sos_index
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index

        self.word_positional_encoder = PositionalEncoding(self.d_model, max_len=self.max_output_len)

        self.embedding  = InputEmbeddings(self.d_model, self.vocab_size)
        self.y_mask = generate_square_subsequent_mask(self.max_output_len) # 上三角掩码
        self.word_positional_encoder = PositionalEncoding(self.d_model, max_len=self.max_output_len)
        # self.transformer_decoder  = Decoder(config.d_model, build_transformer_decoder(config.d_model, config.num_decoder_layers, config.heads,config.dropout,config.dim_feedforward))
        # self.transformer_decoder = TransformerDecoderLayer(d_model=config.d_model, nhead=config.heads, dim_feedforward=config.dim_feedforward, dropout=config.dropout)
        torch.manual_seed(42)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.heads, dim_feedforward=self.dim_feedforward, dropout=self.decoder_dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, self.num_decoder_layers)
        self.fc = nn.Linear(self.d_model, self.vocab_size)


    def forward(self, y, encoded_x):

        # Convert the labels to a format suitable for embedding
        y = y.permute(1, 0)
        # Convert each token ID in the labels to its corresponding embedding vector. 
        # Embedding vectors reflect the semantic relationships between symbols represented by different IDs. 
        # Multiply by the square root of d_model to ensure the variance of the embedding matrix is 1. 
        # Output shape: (Sy, batch_size, d_model)
        y = self.embedding(y)
        # Embed positional encoding
        y = self.word_positional_encoder(y)
        Sy = y.shape[0]
        # Generate an Sy×Sy upper triangular mask where the lower-left part is 0 and the upper-right part is -inf.
        # Its main role is to mask future information.
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)
        # Perform forward propagation through the decoder module. 
        # Output shape: (Sy, batch_size, d_model)
        # output = self.transformer_decoder(y.permute(1,0,2), encoded_x, src_mask=None, tgt_mask=y_mask)
        output = self.transformer_decoder(y, encoded_x, y_mask)
        # Pass through a linear layer to get probability predictions for each token. 
        # Output shape: (Sy, batch_size, num_classes)
        output = self.fc(output)

        return output
        

class ViT_Encoder(nn.Module):
    def __init__(self,
                 config):
        '''
        :param image_size: 光谱图像的尺寸
        :param patch_size: 切割后的尺寸
        :param dim: embedding_dim
        :param num_encoder_layers:  transformer的层数
        :param heads:
        :param channels:
        :param dim_head:
        :param dropout:
        :param emb_dropout:
        '''
        super().__init__()
        image_height, image_width = pair(tuple(config.model_cfg.image_size))
        patch_height, patch_width = pair(tuple(config.model_cfg.patch_size))

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.model_cfg.channels * patch_height * patch_width
        self.d_model = config.model_cfg.d_model
        self.emb_dropout = config.model_cfg.emb_dropout
        self.vocab_size = config.vocab_size 
        self.heads = config.model_cfg.heads
        self.dim_feedforward = config.model_cfg.dim_feedforward 
        self.encoder_dropout = config.model_cfg.encoder_dropout
        self.num_encoder_layers = config.model_cfg.num_encoder_layers

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, self.d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 10, self.d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.dropout = nn.Dropout(self.emb_dropout)
        self.type_embedding = InputEmbeddings(self.d_model, self.vocab_size)
        self.type_linear = nn.Linear(3, 10)
        # self.transformer_encoder = Encoder(config.d_model, build_transformer_encoder(config.d_model, config.num_encoder_layers, config.heads,config.dropout,config.dim_feedforward))
        torch.manual_seed(42)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.heads, dim_feedforward=self.dim_feedforward, dropout=self.encoder_dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, self.num_encoder_layers)

    def encode(self, img,t=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = t
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + cls_tokens.shape[1])]
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        return x

    def forward(self,w,d):

        w = self.type_embedding(w)
        w = self.type_linear(w.permute(0, 2, 1))  # 调整为 [128, 128, 3]
        w = w.permute(0, 2, 1)
        # t = t[:, 1:2, :]
        x = self.encode(d.unsqueeze(1).unsqueeze(-1),w)
        # conbine = torch.cat((x, t), dim=1)
        # x = self.fc_transformencoder(conbine)
        x = x.permute(1, 0, 2)

        return x  # (x,batch_size,d_model)


class Optical_GPT(nn.Module):
    def __init__(
            self,config) -> None:
        super().__init__()
        self.max_output_len = config.trainer_cfg.max_output_len + 2
        self.sos_index = config.sos_index
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        self.beam_size = config.trainer_cfg.beam_size
        self.temperature = config.trainer_cfg.temperature
        if config.trainer_cfg.encoder_model == "ViT":
            self.inverse_encoder = ViT_Encoder(config)
        elif config.trainer_cfg.encoder_model == "Transformer":
            self.inverse_encoder = Inverse_Encoder(config)
        elif config.trainer_cfg.encoder_model == "CNN":
            self.inverse_encoder = CNN_Encoder(config)
        else:
            raise ValueError("Invalid encoder model name, your input should be 'ViT', 'Transformer' or 'CNN'.")
        self.inverse_decoder = Inerse_Decoder(config)


    def forward(self, w: Tensor,d:Tensor, o: Tensor = None) -> Tensor:
        """
        inverse : d: wave_data, w: words, o: para_output
        """
        if o is None:
            return self.predict(w, d)
        encoded_x = self.inverse_encoder(w,d)
        output = self.inverse_decoder(o, encoded_x)
        output = output.permute(1,2,0)
        return output

    
    def predict(self, w: Tensor, d: Tensor, beam_size=3, temperature=1.0) -> Tensor:
        batch_size = w.shape[0]
        encoded_x = self.inverse_encoder(w, d)
        output_indices = torch.full((batch_size, self.max_output_len), self.pad_index).type_as(w).long()
        output_indices[:, 0] = self.sos_index

        beams = [(output_indices[:, :1], torch.zeros(batch_size).type_as(w))]  # [(seq, score)]

        for Sy in range(1, self.max_output_len):
            candidates = []
            for seq, score in beams:
                logits = self.inverse_decoder(seq, encoded_x) / temperature
                probs = F.softmax(logits[-1], dim=-1)
                topk_probs, topk_indices = probs.topk(beam_size, dim=-1)

                for k in range(beam_size):
                    new_seq = torch.cat([seq, topk_indices[:, k:k+1]], dim=-1)
                    new_score = score + topk_probs[:, k].log() 
                    candidates.append((new_seq, new_score))

            candidates = sorted(candidates, key=lambda x: x[1].sum(), reverse=True)
            beams = candidates[:beam_size]

            if all((seq[:, -1] == self.eos_index).all() for seq, _ in beams):
                break

        best_seq, _ = beams[0]
        eos_positions = first_element(best_seq, self.eos_index)
        for i in range(batch_size):
            j = int(eos_positions[i].item()) + 1
            best_seq[i, j:] = self.pad_index

        return best_seq
    

