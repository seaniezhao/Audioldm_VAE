import torch

def check_state_dict(ckpt_path, device=None):
   
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")


    state_dict = torch.load(ckpt_path, map_location=device)

    # state_dict = checkpoint["state_dict"]
    print("Model's state_dict:")
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())


if __name__ == "__main__":
    check_state_dict("/home/sean/checkpoints/audioldm2/first_stage.pt")