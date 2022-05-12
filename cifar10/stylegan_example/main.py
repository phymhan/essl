import torch
from stylegan_model import Generator, Encoder

g_ckpt = 'path/to/g_ckpt'
device='cuda'
image_size = 32

g_ckpt = torch.load(g_ckpt, map_location=device)
latent_dim = g_ckpt['args'].latent

generator = Generator(image_size, latent_dim, 8).to(device)
generator.load_state_dict(g_ckpt["g_ema"], strict=False)
generator.eval()
print('[generator loaded]')

truncation = 0.7
trunc = generator.mean_latent(4096).detach().clone()

with torch.no_grad():
    noise = torch.randn(8*8, latent_dim, device=device)
    latent = generator.get_latent(noise)
    imgs_gen, _ = generator([latent],
                            truncation=truncation,
                            truncation_latent=trunc,
                            input_is_latent=True,
                            randomize_noise=True)

    imgs_gen = torch.clamp(imgs_gen, -1, 1)
    # NOTE: images are in range [-1, 1]
