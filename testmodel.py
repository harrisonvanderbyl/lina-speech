import wave
import numpy as np
from model.lina import Lina
from model.tools import delay_rvq
from vocos import Vocos
from einops import repeat
from transformers import PreTrainedTokenizerFast
from inference import *


config = "rwkv6_60.yaml"
model_state_dict = "rwkv6_60_libritts.pt"
tokenizer_file = "bpe256.json"

model = instantiate_load(config, model_state_dict).eval().to("cuda")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz")

#txt = "Obviously I'm fast, and I'm going to have to utilize that speed, but I'm also going to have to be smart. I'm also strong, and I have a punch, so I'm going to have to use all my qualities to be able to fight against Angulo, and I just feel whoever's the best prepared that night is going to win."
# txt = "Discovery is a child's privilege. I mean the small child, the child who is not afraid to be wrong, to look silly, to not be serious, and to act differently from everyone else. He is also not afraid that the things he is interested in are in bad taste or turn out to be different from his expectations, from what they should be, or rather he is not afraid of what they actually are. He ignores the silent and flawless consensus that is part of the air we breathe, the consensus of all the people who are, or are reputed to be, reasonable."
txt = "The analogy that came to my mind is of immersing the nut in some softening liquid, and why not simply water. From time to time you rub so the liquid penetrates better, and otherwise you let time pass. The shell becomes more flexible through weeks and months, when the time is ripe, hand pressure is enough, the shell opens like a perfectly ripened avocado. "

txt = "[BOS]" + txt + "[EOS]"
txt = torch.LongTensor(tokenizer.encode(txt))

qs, atts, stop_tokens, cuts = model.generate_batch(
                            txt,
                            batch_size=4,
                            k=0.8, #topk sampling
                            temp=2.0, #temperature sampling
                            max_seqlen=2000, #maximum sequence length - this is the maximum checkpoints have been trained on
                            first_greedy_quant=0 , #first residual quantizer to be greedy-sampled
                            device="cuda",
)
bandwidth_id = torch.tensor(1)

print("Syntheses :")
for i, cut in enumerate(cuts):
    feat = vocos.codes_to_features(cut[0].cpu())
    wav = vocos.decode(feat, bandwidth_id=bandwidth_id)
    
    # save wav
    wav = wav.cpu().numpy()
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    with wave.open(f"{i}output.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(wav.tobytes())
    2000