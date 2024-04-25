import wave
import numpy as np
from model.lina import Lina
from model.tools import delay_rvq
from vocos import Vocos
from einops import repeat
from transformers import PreTrainedTokenizerFast
from inference import *
import pyaudio

config = "rwkv6_60.yaml"
model_state_dict = "rwkv6_60_libritts.pt"
tokenizer_file = "bpe256.json"

model = instantiate_load(config, model_state_dict).eval().to("cuda")
#prepare the audio prompt
prompt = torch.load("prompt/spk2.pt")
prompt_codec = delay_rvq(prompt["encodec"] + model.n_special_token_in, head_token=1, tail_token=2) #delaying scheme as introduced by MusicGen
prompt_codec = prompt_codec[:model.n_quant,:-10].unsqueeze(1) #trailing a short amount in order to avoid codec side-effect

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

# txt = "Obviously I'm fast, and I'm going to have to utilize that speed, but I'm also going to have to be smart. I'm also strong, and I have a punch, so I'm going to have to use all my qualities to be able to fight against Angulo, and I just feel whoever's the best prepared that night is going to win."
txt = "Discovery is a child's benefit. I mean the small child, the child who is not afraid to be wrong, to look silly, to not be serious, and to act differently from everyone else. He is also not afraid that the things he is interested in are in bad taste or turn out to be different from his expectations, from what they should be, or rather he is not afraid of what they actually are. He ignores the silent and flawless consensus that is part of the air we breathe, the consensus of all the people who are, or are reputed to be, reasonable."
# txt = "How we doing everybody! Welcome to the show! I'm your host, the one and only, Harrison Vanderbyl."

txt = "[BOS]" +" "+ txt + "[EOS]"
txt = torch.LongTensor(tokenizer.encode(txt))

qs, atts, stop_tokens, cuts = model.generate_batch(
                            txt,
                            batch_size=1,
                            k=0.5, #topk sampling
                            temp=1.0, #temperature sampling
                            max_seqlen=2000, #maximum sequence length - this is the maximum checkpoints have been trained on
                            first_greedy_quant=0 , #first residual quantizer to be greedy-sampled
                            # prompt=prompt_codec, #provides source codec
                            device="cuda"
)
bandwidth_id = torch.tensor(0)

print("Syntheses :")
# for i, cut in enumerate(cuts):
    # feat = vocos.codes_to_features(cut[0].cpu())#[...,prompt_codec.shape[2]:])
    # wav = vocos.decode(feat, bandwidth_id=bandwidth_id)
    
    # save wav
    # wav = wav.cpu().numpy()
    # wav = np.clip(wav, -1, 1)
    # wav = (wav * 32767).astype(np.int16)
    
    # for bytestream in wav:
    #     stream.write(bytestream)
    # with wave.open(f"{i}output.wav", "wb") as f:
    #     f.setnchannels(1)
    #     f.setsampwidth(2)
    #     f.setframerate(24000)
    #     f.writeframes(wav.tobytes())