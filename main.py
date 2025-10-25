import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

from dataset import SegDataset
from config import Config


# ===============================================================
# 1. Training Function
# ===============================================================
def main(cfg_path: str = "defaults.json"):
    # 設定読み込み
    cfg = Config.from_json(cfg_path)
    device = torch.device("cuda")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================================
    # 2. Dataset / DataLoader
    # ===========================================================
    dataset = SegDataset(
        img_dir=cfg.dataset.img_dir,
        seg_dir=cfg.dataset.seg_dir,
        img_suffix=cfg.dataset.img_suffix,
        seg_suffix=cfg.dataset.seg_suffix,
        size=cfg.dataset.size,
        num_classes=cfg.dataset.num_classes,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # ===========================================================
    # 3. モデル構築 (StableDiffusion + ControlNet)
    # ===========================================================
    base_model_id = cfg.model.model_id
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)

    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model_id, subfolder="scheduler"
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.unet.train()
    pipe.controlnet.train()

    if is_xformers_available():
        pipe.unet.enable_xformers_memory_efficient_attention()

    # ===========================================================
    # 4. Optimizer & AMP準備
    # ===========================================================
    optimizer = AdamW(pipe.controlnet.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device=device, enabled=True)

    # ===========================================================
    # 5. Training Loop
    # ===========================================================

    # 最初に空文字列をエンコードしておく
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        ehs = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    for epoch in range(cfg.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            seg = batch["conditioning"].to(device, non_blocking=True)
            bsz = pixel_values.size(0)

            # === VAE encode (no grad) ===
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad()

            # === Forward with autocast ===
            with torch.amp.autocast(device_type=str(device), enabled=True):
                # ControlNet + UNet Forward
                down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=ehs.repeat(bsz, 1, 1),
                    controlnet_cond=seg,
                    return_dict=False,
                )

                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=ehs.repeat(bsz, 1, 1),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # Loss & Backprop
                loss = torch.nn.functional.mse_loss(model_pred, noise)

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Save checkpoint
        save_path = out_dir / f"controlnet_epoch{epoch + 1}.pt"
        torch.save(controlnet.state_dict(), save_path)
        print(f"Saved: {save_path.as_posix()}")

    print("Training complete.")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main("./configs/defaults.json")
