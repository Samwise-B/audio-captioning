import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, ds, i: int, max_len: int):
    tokeniser = ds.tk
    tk_to_id = ds.tk_to_id

    with torch.inference_mode():
        # idx = random.randint(0, ds.__len__())
        (audio, masks, _, cap_targ, _) = ds[i]
        audio, masks = audio.to(device), masks.to(device)
        cap_targ = cap_targ.to(device)
        inpt = torch.tensor(
            [tk_to_id("<|startoftranscript|>")]
            + [tk_to_id("<|en|>")]
            + [tk_to_id("<|transcribe|>")]
            + [tk_to_id("<|notimestamps|>")],
            device=device,
        ).unsqueeze(0)
        for _ in range(cap_targ.shape[-1]):
            out = model(audio, masks, inpt)
            pred = out.logits
            new_logit = pred[:, -1, :]
            next_token = torch.argmax(new_logit, dim=-1)
            inpt = torch.cat([inpt, next_token.unsqueeze(0)], dim=-1)

        return tokeniser.decode(
            inpt.squeeze(), skip_special_tokens=False
        ), tokeniser.decode(cap_targ, skip_special_tokens=False)
