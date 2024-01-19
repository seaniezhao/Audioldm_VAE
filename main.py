import torch
from latent_encoder.audioencoder import AutoencoderKL
import soundfile as sf
 
from utils import get_audioldm_48k_config, get_mel_from_wav_file
 

def get_vae_config():
    cfg = get_audioldm_48k_config()
    model_cfg = cfg["model"]
    f_cfg = model_cfg["params"]
    first_stage_params_config = f_cfg["first_stage_config"]["params"]
    return first_stage_params_config


def create_model(checkpoint_path, device=None):
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    params = get_vae_config()
    model = AutoencoderKL(**params)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def main():
  
    mel = get_mel_from_wav_file("train_tgt_3000.flac")
 
    mel = mel.unsqueeze(0).unsqueeze(1)

    model = create_model("/home/sean/checkpoints/audioldm2/first_stage.pt")
    print(model)

    model.eval()
    dec_mel, posterior = model.forward(mel)

    y1 = model.decode_to_waveform(dec_mel)
    print(y1.shape)
    sf.write("test1.flac", y1[0], 48000)

    


if __name__ == "__main__":
    
    main()

    