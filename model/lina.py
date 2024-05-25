from time import sleep
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import EinMix
from model.attentive_rnn import AttentiveRNN
from model.multiembed import MultiEmbedding
from pytorch_lightning.cli import LightningCLI
from torch import Tensor, nn
from transformers import get_cosine_schedule_with_warmup

torch.multiprocessing.set_sharing_strategy("file_system")
from .accuracy import MulticlassAccuracy
from .tools import sequence_mask, topk_sampling, undelay_rvq
from vocos import Vocos

from threading import Thread
import pyaudio
pya = pyaudio.PyAudio()
bytedata = []
lastbits = bytes()
pointer = 0
def callback(in_data, frame_count, time_info, status):
        global bytedata, lastbits, pointer
        
        # if lastbits.__len__() < frame_count*2:
        #     while bytedata.__len__() == 0:
        #         sleep(0.1)
        #     mdata = bytedata.pop(0).numpy().tobytes()
        #     lastbits += mdata
        while lastbits.__len__() < pointer + frame_count*2:
            sleep(0.01)
            
            
            
            
        # print(frame_count)
        
        # byte data to frames
        data = lastbits[pointer:pointer + frame_count*2]
        pointer = min(pointer + frame_count*2, lastbits.__len__())
        
        if(data.__len__() < frame_count*2):
            data = data + (b'\x00' * (frame_count*2 - data.__len__()))
            
        # If len(data) is less than requested frame_count, PyAudio automatically
        # assumes the stream is finished, and the stream stops.
        return (data, pyaudio.paContinue)
    
stream = pya.open(format=pya.get_format_from_width(width=2), channels=1, rate=24000, output=True, stream_callback=callback)
# stream.start_stream()
started = False
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
n_special_token_in_g = 2

n_quant_g = 4
streambuffer = []
running = True
lasttoks = []
def submit(mnta, buffsize, index):
    global lasttoks, n_special_token_in_g, lastbits
    index += buffsize
    # if oldtoks.__len__() == 0:
    #     oldtoks = mnta.copy()
        
    #     continue
    if lasttoks.__len__() == 0:
        lasttoks = mnta.copy()
        return index
        
        
    # mnt = list(mnt.__reversed__())
    # if mnt.__len__() > 1:
    
    mnt = torch.stack(lasttoks+mnta, dim=2).squeeze(-1)
    # oldtoks = lasttoks.copy()
    lasttoks = mnta.copy()
    
    mnt = undelay_rvq(mnt)
    mnt = mnt.sub(n_special_token_in_g).clamp_min(0)
    # print(n_special_token_in_g)
    mnt = mnt[:,0,0:buffsize].squeeze(0)
    
    # else :
    #     mnt = torch.stack(mnt[0],1).reshape(1,-1,1).sub(n_special_token_in_g).relu()
    feat = vocos.codes_to_features(mnt)#[...,prompt_codec.shape[2]:])
    wav:torch.Tensor = vocos.decode(feat, bandwidth_id=torch.tensor(0))
    wav = wav.clamp(-1, 1)
    wav = (wav * 32767).to(torch.int16)
    # stream.write(wav.cpu().numpy().tobytes())
    lastbits += wav.cpu().numpy().tobytes()
    return index

def outloop():
    global streambuffer,bytedata,started, lastbits
    
    index = 0
    
    buffsize = 64
    while running or streambuffer.__len__() >= index+buffsize:
        if streambuffer.__len__() >= index+buffsize:
            
            mnta = streambuffer[index:index + buffsize].copy()
            index = submit(mnta, buffsize, index)
            # if not started:
            #     stream.start_stream()
            #     started = True
        else:
            sleep(0.1)
    
    submit(streambuffer[index:], streambuffer.__len__()-index, index)
    sleep(2)
            
t = Thread(target=outloop)
t.start()

def endstream():
    global running
    running = False

def exists(x):
    return x is not None


class Lina(ptl.LightningModule):
    def __init__(
        self,
        attentive_rnn: AttentiveRNN,
        d_model: int,
        quant_layer: List[int],
        n_codebook: int,
        n_special_token_in: int,
        n_special_token_out: int,
        n_txt_vocab: int,
        tie_embed: bool = False,
        d_context: Optional[int] = None,
        txt_encoder: Optional[nn.Module] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        n_warmup_steps: int = 500,
        n_training_steps: int = 300000,
    ):
        global n_special_token_in_g, n_quant_g
        super(Lina, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps

        self.n_quant = len(quant_layer)
        self.n_codebook = n_codebook
        self.n_special_token_in = n_special_token_in
        n_special_token_in_g = self.n_special_token_in
        n_quant_g = self.n_quant
        self.n_special_token_out = n_special_token_out
        self.n_txt_vocab = n_txt_vocab
        self.n_target_vocab = n_codebook + n_special_token_out  # no padding token

        self.txt_encoder = txt_encoder

        self.txt_embed = nn.Embedding(
            n_txt_vocab,
            d_context if exists(d_context) else d_model,
            padding_idx=0,
        )
        self.rvq_embed = MultiEmbedding(self.n_quant, n_codebook + n_special_token_in, d_model, padding_idx=0) 

        self.attentive_rnn = attentive_rnn

        self.logits_head = EinMix(
            "b n d -> b n q l",
            weight_shape="q l d",
            q=self.n_quant,
            d=d_model,
            l=self.n_target_vocab,
        )
        if tie_embed:
            self.logits_head.weight = self.rvq_embed.weight

        self.save_hyperparameters()

        self.accuracy_metric = MulticlassAccuracy(
            self.n_target_vocab,
            top_k=10,
            ignore_index=[0, 1],
        )

    def step(self, x, y, x_lens, y_lens):
        # b: batch
        # n: seq length
        # q: quantizers
        # d: model dim
        # l: target vocab
        x_embd = self.txt_embed(x)
        y_embd = self.rvq_embed(rearrange(y, "b n q -> q b n"))
        q, b, n, d = y_embd.shape
        y_embd = reduce(y_embd, "q b n d -> b n d", "sum", q=q)

        x_mask, y_mask = map(sequence_mask, (x_lens, y_lens))
        
        x_enc = self.txt_encoder(x_embd, x_mask) if exists(self.txt_encoder) else x_embd
        
        y_hat, att = self.attentive_rnn(
            y_embd[:, :-1, :],
            x_enc,
            y_mask[:, 1:],
            x_mask,
        )
        logits = self.logits_head(y_hat)
        masked_logits = logits[y_mask[:, 1:], :, :]
        masked_target = y[:, 1:][y_mask[:, 1:], :]
        flat_logits = rearrange(masked_logits, "n q l -> (n q) l")
        flat_target = rearrange(masked_target, "n q   -> (n q)")

        loss = F.cross_entropy(flat_logits, flat_target)

        return logits, loss, att, masked_logits, masked_target

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_lens: Tensor,
        y_lens: Tensor,
    ):
        
        logits, loss, att, masked_logits, masked_target = self.step(x, y, x_lens, y_lens)
        with torch.no_grad():
            accs = [
                self.accuracy_metric(
                    masked_logits[:, i],
                    masked_target[:, i],
                )
                for i in range(self.n_quant)
            ]

        return logits, loss, att, accs

    @torch.inference_mode()
    def generate_batch(
        self,
        x: Tensor,
        batch_size: int=3,
        prompt: Optional[Tensor] = None,
        device: str = "cpu",
        max_seqlen: int = 1000,
        k: float = 0.0,
        first_greedy_quant = None,
        temp: float = 1.0,
        state = None
    ):
        x = repeat(x, "n -> b n", b=batch_size).to(device)
        stop_token = torch.ones(self.n_quant, 1, 1, device=device) * 2
        y_start = torch.ones(self.n_quant, batch_size, 1, device=device).long()

        x_embd = self.txt_embed(x)
        
        y_embd = reduce(self.rvq_embed(y_start), "q b n d -> b n d", "sum")
        
        p_len = -1
        if exists(prompt):
            if prompt.shape[1] != batch_size:
                prompt = repeat(prompt, "q 1 n -> q b n", b=batch_size)
            prompt = reduce(self.rvq_embed(prompt.to(device)), "q b n d -> b n d", "sum")
            p_len = prompt.shape[1]


        x_mask = torch.ones_like(x, device=device).bool()

        x_enc = self.txt_encoder(x_embd, x_mask) if exists(self.txt_encoder) else x_embd
        print("Encoding done")
        if state == None:
            self.attentive_rnn.init_state(max_seqlen=max_seqlen)
        else:
            self.attentive_rnn.set_state(state)
        print("Init done")
        qs, atts, stop_tokens = [], [], []

        for t in range(max_seqlen):
            y_embd, att = self.attentive_rnn.step(y_embd, x_enc, time_step=t)
            atts.append(att)
            logits = self.logits_head(y_embd)
            logits = rearrange(logits, "b 1 q l -> q b l")
            q_sampled = []
            qmax = []
            for i, q in enumerate(logits):
                q_sampled.append(
                    topk_sampling(q, k=k, greedy=True)
                    if first_greedy_quant and i > first_greedy_quant
                    else topk_sampling(q, k=k, greedy=False, temp=temp)
                )
                qmax.append(topk_sampling(q, k=1, greedy=True))
            q_sampled = torch.stack(q_sampled)
            qmax = torch.stack(qmax)
            qs.append(q_sampled)

            is_stop_token = (qmax == stop_token).prod(dim=0)
            ois_stop_token = (q_sampled == stop_token).prod(dim=0)
            stop_tokens.append(is_stop_token)
            if is_stop_token.prod() or ois_stop_token.prod():
                break

            if exists(prompt) and t < p_len:
                y_embd = prompt[:, [t]]
            else:
                y_embd = self.rvq_embed(q_sampled)
                y_embd = reduce(y_embd, "q b n d -> b n d", "sum")
            
            if t > self.n_quant:
                streambuffer.append(qs[-1].cpu())

        atts =  torch.cat(atts, dim=2) if exists(atts[0]) else None
        qs = torch.stack(qs, dim=2).squeeze(-1)
        stop_tokens.append(torch.ones(batch_size, 1, device=device))
        stop_tokens = torch.stack(stop_tokens, dim=1).squeeze(-1)
        b, n = stop_tokens.shape
        rvq = (undelay_rvq(qs) - self.n_special_token_in).clamp_min(0)
        stop_idx = (stop_tokens * rearrange(torch.arange(n, device=stop_tokens.device), "n -> 1 n")).long()
        cuts = []
        for i in range(stop_idx.shape[0]):
            idx = torch.unique(stop_idx[i])[1]
            cuts.append((rvq[:,[i], :idx-self.n_quant], atts[i, 1, :idx]))
        return qs, atts, stop_tokens, cuts, self.attentive_rnn.get_state()

       
    @torch.inference_mode()
    def generate(
        self,
        x: Tensor,
        teach_force: Optional[Tensor] = None,
        prompt: Optional[Tensor] = None,
        device: str = "cpu",
        max_seqlen: int = 1000,
        k: int = 100,
        first_greedy_quant: int = 1,
        temp: float = 1.0,
    ):
        x = rearrange(x, "n -> 1 n").to(device)
        y_start = torch.ones(self.n_quant, 1, 1, device=device).long()

        x_embd = self.txt_embed(x).to(device)
        y_embd = reduce(self.rvq_embed(y_start), "q b n d -> b n d", "sum")
        
        p_len = -1
        if exists(prompt):
            prompt = reduce(self.rvq_embed(prompt), "q b n d -> b n d", "sum")
            p_len = prompt.shape[1]


        x_mask = torch.ones_like(x, device=device).bool()

        x_enc = self.txt_encoder(x_embd, x_mask) if exists(self.txt_encoder) else x_embd

        self.attentive_rnn.init_state(max_seqlen=max_seqlen)

        qs = []
        atts = []

        for t in range(max_seqlen):
            y_embd, att = self.attentive_rnn.step(y_embd, x_enc, time_step=t)
            atts.append(att)
            logits = self.logits_head(y_embd)
            logits = rearrange(logits, "b 1 q l -> q b l")
            q_sampled = []

            for i, q in enumerate(logits):
                q_sampled.append(
                    topk_sampling(q, k=k, temp=temp)
                    if i < first_greedy_quant
                    else topk_sampling(q, k=1)
                )

            q_sampled = torch.stack(q_sampled)
            qs.append(q_sampled.squeeze())
            
            #hacky
            if q_sampled.squeeze().prod() == 2**self.n_quant:
                break

            if exists(prompt) and t < p_len:
                y_embd = prompt[:, [t]]
            else:
                y_embd = reduce(self.rvq_embed(q_sampled), "q b n d -> b n d", "sum")

        atts =  torch.cat(atts, dim=2) if exists(atts[0]) else None
        return qs, atts
    #def on_train_epoch_start(self):
    #    self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)


    def training_step(self, batch, _):
        x, y, x_len, y_len, code, phon, _ = batch
        logits, loss, att, accs = self(x, y, x_len, y_len)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        for i, acc in enumerate(accs):
            self.log("train_acc_" + str(i), acc, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, idx):
        x, y, x_len, y_len, code, phon, _ = batch
        logits, loss, att, accs = self(x, y, x_len, y_len)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for i, acc in enumerate(accs):
            self.log("val_acc_" + str(i), acc, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        params = [
            {
                "params": self.parameters(),
                "weight_decay": self.weight_decay,
            }
        ]
        opt = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=self.betas,
        )

        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps,)

        return [opt], [{'scheduler': scheduler, "interval": "step"}]

def cli():
    class LinaCli(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "data.init_args.quant_layer", "model.init_args.quant_layer"
            )

    LinaCli(parser_kwargs={"parser_mode": "omegaconf"}, save_config_kwargs={"overwrite": True})
