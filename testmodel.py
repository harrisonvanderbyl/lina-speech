import wave
import numpy as np
from model.lina import Lina,endstream
from model.tools import delay_rvq
from vocos import Vocos
from einops import repeat
from transformers import PreTrainedTokenizerFast
from inference import *
import pyaudio
dd = "rwkv6"
# dd = "rwkv6"
config = dd+"_60.yaml"
model_state_dict = dd+"_60_libritts.pt"
tokenizer_file = "bpe256.json"

model = instantiate_load(config, model_state_dict).eval().to("cuda")
#prepare the audio prompt
prompt = torch.load("prompt/spk2.pt")
prompt_codec = delay_rvq(prompt["encodec"] + model.n_special_token_in, head_token=1, tail_token=2) #delaying scheme as introduced by MusicGen
prompt_codec = prompt_codec[:model.n_quant,:-10].unsqueeze(1) #trailing a short amount in order to avoid codec side-effect

tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

# txt = "Obviously I'm fast, and I'm going to have to utilize that speed, but I'm also going to have to be smart. I'm also strong, and I have a punch, so I'm going to have to use all my qualities to be able to fight against Angulo, and I just feel whoever's the best prepared that night is going to win."
# txt = "Discovery is a child's benefit. I mean the small child, the child who is not afraid to be wrong, to look silly, to not be serious, and to act differently from everyone else. He is also not afraid that the things he is interested in are in bad taste or turn out to be different from his expectations, from what they should be, or rather he is not afraid of what they actually are. He ignores the silent and flawless consensus that is part of the air we breathe, the consensus of all the people who are, or are reputed to be, reasonable."
txt = """Raine Talis, no longer of the Gosruk Guardians, woke with a start.

The last thing she remembered was lying down in one of Catos odd cots, planning how she was going to get through the portal while the other her lured out the Bismuth.

Yet she was not in a cot, but rather clothed and armored and strapped into a familiar seat — the contraption Cato had used to bring her back down to Sydea.

She glanced around, finding Leese and Dyen stirring in the other seats.  Reflexively she tried to bring up the System, but nothing happened, confirming her conclusion.

She was the other her.  All the plans shed been considering no longer applied; no matter what, she wasnt going through that portal.

Instead it was up to her to somehow convince a Bismuth to stray away from the populated regions, out to where Cato could take care of him. And possibly to die in the process.

Suddenly she actually understood all of Catos warnings. Her tail twitched uncontrollably, caught in a welter of different emotions ranging from abyssal regret to a crushing claustrophobic panic about a future now set in stone. A strange resentment or jealousy stole into her, directed at herself.

Both her past decisions and the other her, existing somewhere down on Sydea, which got to keep and benefit from all the work she had done.

'Good, youre all awake,' Catos voice came from a Sydean version of himself piloting the craft.

Raine shook herself away from her strange inward spiral, taking a deep breath and letting it out.

Beside her, Leese pulled herself from similar contemplations, but when she glanced back at Dyen, he didnt seem bothered."""

# txt = "[BOS]"+""+ txt + "[EOS]"
# txt = txt.split("[split]")
# fhalf,txt = txt[0],txt[1]
# fhalf, txt = fhalf+"[EOS]","[BOS]"+ txt
# txt2 = torch.LongTensor(tokenizer.encode(fhalf))
state = None

for num,i in enumerate(txt.split("\n\n")):
    i = prompt["transcription"] + i if num==-1 else i
    txt2 = torch.LongTensor(tokenizer.encode("[BOS][BOS] ,. "+i+"[EOS]"))
    qs, atts, stop_tokens, cuts,stateo = model.generate_batch(
                                txt2,
                                batch_size=1,
                                k=30, #topk sampling
                                temp=0.8, #temperature sampling
                                max_seqlen=2000, #maximum sequence length - this is the maximum checkpoints have been trained on
                                # first_greedy_quant=2, #first residual quantizer to be greedy-sampled
                                # prompt=prompt_codec if num==-1 else None, #audio prompt
                                device="cuda",
                                state=state
)
    state = stateo if state is None else state



endstream()
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