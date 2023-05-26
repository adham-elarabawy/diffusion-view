import torch
import numpy as np
from tqdm import tqdm


# def preprocess(image):
#     w, h = image.size
#     w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
#     image = image.resize((w, h), resample=Image.LANCZOS)
#     image = np.array(image).astype(np.float32) / 255.0
#     image = image[None].transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     return 2.0 * image - 1.0

def show_lat(latents, pipe):
    # utility function for visualization of diffusion process
    with torch.no_grad():
        images = pipe.decode_latents(latents)
        print("Image statistics: ", images.mean(), images.std(), images.min(), images.max())
        im = pipe.numpy_to_pil(images)[0].resize((128, 128))
    return im

def null_text_inversion(
        pipe,
        all_latents,
        prompt,
        num_opt_steps=10,
        lr=0.01,
        tol=1e-5,
        guidance_scale=7.5,
        eta: float = 0.0,
        generator=None,
        T=50,
        negative_prompt=None
):
    # get null text embeddings for prompt
    null_text_prompt = ""
    null_text_input = pipe.tokenizer(
        null_text_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    null_text_embeddings = torch.nn.Parameter(pipe.text_encoder(null_text_input.input_ids.to(pipe.device))[0],
                                              requires_grad=True)
    null_text_embeddings = null_text_embeddings.detach()
    null_text_embeddings.requires_grad_(True)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        [null_text_embeddings],  # only optimize the embeddings
        lr=lr,
    )

    # step_ratio = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    text_embeddings = pipe._encode_prompt(prompt, pipe.device, 1, False, negative_prompt).detach()
    # input_embeddings = torch.cat([null_text_embeddings, text_embeddings], dim=0)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    all_null_texts = []
    latents = all_latents[-1]
    latents = latents.to(pipe.device)

    pipe.scheduler.set_timesteps(T)
    for timestep, prev_latents in pipe.progress_bar(zip(pipe.scheduler.timesteps, reversed(all_latents[:-1])), total=T):
        prev_latents = prev_latents.to(pipe.device).detach()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep).detach()
        noise_pred_text = pipe.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample.detach()

        for idx in range(num_opt_steps):
            # predict the noise residual
            noise_pred_uncond = pipe.unet(latent_model_input, timestep,
                                          encoder_hidden_states=null_text_embeddings).sample

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            prev_latents_pred = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample
            loss = torch.nn.functional.mse_loss(prev_latents_pred, prev_latents).mean()
            
            # breakpoint()

            loss.backward()
            # print(idx, loss, null_text_embeddings.grad.mean())
            optimizer.step()
            optimizer.zero_grad()
            if loss < tol:
                break
        all_null_texts.append(null_text_embeddings.detach().cpu().unsqueeze(0))
        latents = prev_latents_pred.detach()
    return all_latents[-1], torch.cat(all_null_texts)

def preprocess_image(image, torch_dtype):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch_dtype) / 127.5 - 1.0
    return image

@torch.no_grad()
def ddim_inversion(pipe, prompt, image, T, generator=None, negative_prompt="", w=1, torch_dtype=torch.float16):
    """
    DDIM based inversion of image to noise

    :param pipe: Diffusion Pipeline
    :param prompt: initial prompt
    :param image: input image that should be inversed
    :param T: num_steps of Diffusion
    :param generator: noise generator
    :param negative_prompt: negative prompt for guidance
    :param w: guidance scale
    :return: initial trajectory
    """
    pp_image = preprocess_image(image, torch_dtype)
    latents = pipe.vae.encode(pp_image.to(pipe.device)).latent_dist.sample(generator=generator) * 0.18215


    context = pipe._encode_prompt(prompt, pipe.device, 1, False, negative_prompt)
    pipe.scheduler.set_timesteps(T)

    next_latents = latents
    all_latents = [latents.detach().cpu().unsqueeze(0)]

    for timestep, next_timestep in zip(reversed(pipe.scheduler.timesteps[1:]),
                                       reversed(pipe.scheduler.timesteps[:-1])):
        latent_model_input = pipe.scheduler.scale_model_input(next_latents, timestep)
        noise_pred = pipe.unet(latent_model_input, timestep, context).sample

        alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_next = pipe.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        f = (next_latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        next_latents = alpha_prod_t_next ** 0.5 * f + beta_prod_t_next ** 0.5 * noise_pred
        all_latents.append(next_latents.detach().cpu().unsqueeze(0))

    return torch.cat(all_latents)

@torch.no_grad()
def reconstruct(pipe, latents, prompt, null_text_embeddings, guidance_scale=7.5, generator=None, eta=0.0, negative_prompt="", T=50):
    text_embeddings = pipe._encode_prompt(prompt, pipe.device, 1, False, negative_prompt)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    latents = latents.to(pipe.device)

    pipe.scheduler.set_timesteps(T)
    for i, (t, null_text_t) in enumerate(pipe.progress_bar(zip(pipe.scheduler.timesteps, null_text_embeddings), total=T)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        input_embedding = torch.cat([null_text_t.to(pipe.device), text_embeddings])
        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    #Post-processing
    image = pipe.decode_latents(latents)
    return image



## RUN

from pathlib import Path
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":

    project_name = "test"
    Path(f"./results/{project_name}").mkdir(parents=True, exist_ok=True)
    device = "cuda:6"

    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # model_id_or_path = "CompVis/stable-diffusion-v1-4"
    scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
    SD_pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, scheduler=scheduler, torch_dtype=torch.float32, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to(device)
    SD_pipe.enable_attention_slicing()
    SD_pipe.enable_xformers_memory_efficient_attention()


    og_image = Image.open("../../data/birk/0.png").resize((512, 512))
    source_prompt = ""
    T = 50

    run_ddim = True
    run_null = True
    run_gen = True

    generator = torch.Generator(device=device)
    if run_ddim:
        print("[Stage 0] Running DDIM Inversion for Initial Trajectory Generation...")
        init_trajectory = ddim_inversion(SD_pipe, source_prompt, og_image, T, generator, torch_dtype=torch.float32)
        print(init_trajectory.shape)
        torch.save(init_trajectory, f"./results/{project_name}/init_trajectory.pt")

        plt.figure(figsize=(20, 8))
        with torch.autocast("cuda"):
            for i, traj in enumerate(init_trajectory[::10]):
                plt.subplot(1, (T // 10) + 1, i + 1)
                plt.imshow(show_lat(traj.to(device), SD_pipe))
                plt.axis("off")
        plt.savefig(f"./results/{project_name}/trajectories.png")

        with torch.inference_mode(), torch.autocast("cuda"):
            z_T = init_trajectory[-1].to(device)
            im = SD_pipe(prompt=source_prompt, latents=z_T, generator=generator)
            im[0][0].save(f"./results/{project_name}/DDIM_reconstruction.png")

    if run_null:
        print("[Stage 1] Running Null-Text-Inversion...")
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        generator = torch.Generator(device=device)
        z_T, null_embeddings = null_text_inversion(SD_pipe, init_trajectory, source_prompt,
                                                    guidance_scale=7.5, generator=generator)

        torch.save(null_embeddings, f"./results/{project_name}/nulls.pt")


    if run_gen:
        print("[Stage 2] Reconstructing + Editing Image via refined Inversion...")
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")

        z_T = init_trajectory[-1]

        SD_pipe.scheduler.set_timesteps(T)
        recon_img = reconstruct(SD_pipe, z_T, source_prompt, null_embeddings, guidance_scale=1)
        plt.imsave(f"./results/{project_name}/reconstructed.png", recon_img[0])

        edited_prompt = ""
        edited_img = reconstruct(SD_pipe, z_T, edited_prompt, null_embeddings, guidance_scale=7.5)
        plt.imsave(f"./results/{project_name}/edited.png", edited_img[0])

        edit_imgs = []
        num_imgs = 10
        for scale in np.linspace(0.5, 10, num_imgs):
            edit_img = reconstruct(SD_pipe, z_T, edited_prompt, null_embeddings, guidance_scale=scale)
            edit_imgs.append(edit_img)

        fig, ax = plt.subplots(1, num_imgs + 1, figsize=(10 * (num_imgs + 1), 10))

        ax[0].imshow(recon_img[0])
        ax[0].set_title("Reconstructed", fontdict={'fontsize': 40})
        ax[0].axis('off')

        for i, scale in enumerate(np.linspace(0.5, 10, num_imgs)):
            ax[i + 1].imshow(edit_imgs[i][0])
            ax[i + 1].set_title("%.2f" % scale, fontdict={'fontsize': 40})
            ax[i + 1].axis('off')

        plt.xlabel(edited_prompt)
        plt.savefig(f"./results/{project_name}/guidance_test.png")