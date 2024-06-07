import sys
import torch
import gradio as gr
from models import Generator
from hparams import get_sampler_hparams
from utils.log_utils import save_images, set_up_visdom, config_log, log, start_training_log, display_images, load_model
from utils.sampler_utils import get_sampler, get_samples, retrieve_autoencoder_components_state_dicts
from PIL import Image
import numpy as np

def resize_image(image, size=(200, 200)):
    return image.resize(size, Image.ANTIALIAS)

def generate_images(H):
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop('embedding.weight')
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    sampler = load_model(sampler, f'{H.sampler}', H.load_step, H.load_dir) # ema or not?
    sampler.n_samples = H.num_samples  # get samples in 5x5 grid  & takes 2100 MiB for one

    images = get_samples(H, generator, sampler)
    print(f"images size : {images.size()}")

    if sampler.n_samples == 1:
        images = torch.clamp(images, 0, 1)
        images_list = images.cpu().permute(0, 2, 3, 1).numpy()
        print(f"images_list size : {images_list.shape}")

        images_list = Image.fromarray((images_list * 255).reshape(256, 256, 3).astype(np.uint8))
        images_list = resize_image(images_list)
        return images_list
    else: # for multiple images

        print(f"when multiple, shape : {images.size()}")
        images = torch.clamp(images, 0, 1)
        images_list = images.cpu().permute(0, 2, 3, 1).numpy()
        print(f"images_list size : {images_list.shape}")

        num_images = images_list.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_images)))
        grid_image = Image.new('RGB', (grid_size * 200, grid_size * 200))

        for idx in range(num_images):
            img = Image.fromarray((images_list[idx] * 255).astype(np.uint8))
            img = resize_image(img)
            grid_x = idx % grid_size
            grid_y = idx // grid_size
            grid_image.paste(img, (grid_x * 200, grid_y * 200))

        final_image = grid_image.resize((grid_size * 256, grid_size * 256), Image.ANTIALIAS)
        return final_image

def main():
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)

    css = """
    .gradio-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .output_image {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .gradio-interface {
        width: 800px;
        height: 800px;
    }
    """

    # description = """
    # ## NAT-Diffuser : Discrete Absorbing Diffusion Meets Neighborhood Attnetion Transformers at Vector-Quantized Space
    # """
    # checkpoint_info = f"""
    # **Checkpoint Name** : {H.sampler}_ema_{H.load_step}.pth
    # **Number of images** : {H.sample_steps}
    # **Number of denoising steps** : 
    # """

    if H.num_samples == 1:
        iface = gr.Interface(fn=lambda: generate_images(H), inputs=[], outputs=gr.Image(type="pil", label="Generated Image", width=256, height=256), css=css,  title="🦁NAT-Diffuser : Generate single 256x256 image!")#, description = description, article = checkpoint_info)
    else:
        iface = gr.Interface(fn=lambda: generate_images(H), inputs=[], outputs=gr.Image(type="pil", label="Generated Image", width=800, height=800, elem_id="output_image"), css=css,  title="🦁NAT-Diffuser: Generate multiple 256x256 images at once!")#, description = description, article = checkpoint_info)

    iface.launch()

if __name__ == '__main__':
    main()
