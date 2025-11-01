import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append('..')
import argparse

import torch

from transformers import UMT5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import export_to_video, load_video

from wan_continuous_transformer import WanTransformer3DModel
from wan_continuous_pipeline import WanContinuousVideoPipeline



@torch.no_grad()
def generate_video(
    prompt: str,
    video_path: str,
    base_model_path: str,
    lora_path: str,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    teacache_treshold: float = 0.0,
    video_height: int = 480,
    video_width: int = 832,
    num_input_frames: int = 24,
    num_output_frames: int = 81,
    negative_prompt: str = "bad quality, worst quality",
    seed: int = 42,
    out_fps: int = 24,
    
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - video_path (str): The video for conditioning.
    - base_model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - teacache_treshold (float): TeaCache value. Best from [0.3, 0.5, 0.7, 0.9].
    - video_height (int): Output video height.
    - video_width (int): Output video width.
    - num_input_frames (int): Input frames count. Last N frames from input video.
    - num_output_frames (int): Output frames count.
    - seed (int): The seed for reproducibility.
    - out_fps (int): FPS of output video.
    """

    # Load the pre-trained Wan2.2 models with the specified precision (bfloat16).
    tokenizer = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    pipe = WanContinuousVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae, 
        scheduler=scheduler,
    )
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload()

    pipe.transformer.load_lora_adapter(
        lora_path,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="video_continuation",
        prefix=None,
    )
    pipe.set_adapters("video_continuation", adapter_weights=1.0)

    previous_video = load_video(video_path)[-num_input_frames:]
    output = pipe(
        previous_video=previous_video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=video_height,
        width=video_width,
        num_frames=num_output_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output_type="pil",

        teacache_treshold=teacache_treshold,
    ).frames[0]
    export_to_video(output, output_path, fps=out_fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan2.1")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="The path of the video for conditioning.",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, help="The path of the LoRA weights to be used")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--video_height", type=int, default=480, help="Output video height")
    parser.add_argument("--video_width", type=int, default=832, help="Output video width")
    parser.add_argument("--num_input_frames", type=int, default=24, help="Input frames count. Last N frames from input video.")
    parser.add_argument("--num_output_frames", type=int, default=81, help="Output frames count")
    parser.add_argument("--negative_prompt", type=str, default="bad quality, worst quality", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--out_fps", type=int, default=24, help="FPS of output video")
    parser.add_argument("--teacache_treshold", type=float, default=0.0, help="TeaCache value. Best from [0.6, 1.0, 1.5]")
    
    args = parser.parse_args()
    
    generate_video(
        prompt=args.prompt,
        video_path=args.video_path,
        base_model_path=args.base_model_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        video_height=args.video_height,
        video_width=args.video_width,
        num_input_frames=args.num_input_frames,
        num_output_frames=args.num_output_frames,
        seed=args.seed,
        out_fps=args.out_fps,
        lora_path=args.lora_path,
        teacache_treshold=args.teacache_treshold,
    )