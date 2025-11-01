import argparse
import os
import threading
import time

import gradio as gr
import torch
from diffusers.utils import export_to_video, load_video
from transformers import UMT5EncoderModel, T5TokenizerFast
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler
)
from datetime import datetime, timedelta

from wan_continuous_transformer import WanTransformer3DModel
from wan_continuous_pipeline import WanContinuousVideoPipeline


os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)


def save_video(tensor, fps):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path, fps=fps)
    return video_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

def main(args):
    tokenizer = T5TokenizerFast.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(args.base_model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(args.base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")

    pipe = WanContinuousVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=args.base_model_path,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae, 
        scheduler=scheduler,
    )
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.enable_model_cpu_offload()
    
    pipe.transformer.load_lora_adapter(
        args.lora_path,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="video_continuation",
        prefix=None,
    )
    pipe.set_adapters("video_continuation", adapter_weights=1.0)

    def infer(
            prompt: str, negative_prompt: str, previous_video: list, num_inference_steps: int, guidance_scale: float, 
            seed: int, width: int, height: int, num_output_frames: int, teacache_treshold: float, progress=gr.Progress(track_tqdm=True)
        ):
        torch.cuda.empty_cache()

        output = pipe(
            previous_video=previous_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_output_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="pil",
        
            teacache_treshold=float(teacache_treshold.value if hasattr(teacache_treshold, 'value') else teacache_treshold),
        ).frames[0]

        return output

    with gr.Blocks() as demo:
        gr.Markdown("""
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                Video Continuation for Wan2.1 Space🤗
                """)

        with gr.Row():
            with gr.Column():
                with gr.Column():
                    video_input = gr.Video(label="Video for conditioning", width=720, height=720)
                    with gr.Row():
                        download_video_button = gr.File(label="📥 Download Video", visible=False)

                prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (Less than 200 Words)", 
                    value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", 
                    placeholder="Enter your prompt here", 
                    lines=3
                )

                with gr.Column():
                    gr.Markdown(
                        "**Optional Parameters**<br>"
                        "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                        "50 steps are recommended for most cases.<br>"
                    )
                    with gr.Row():
                        width = gr.Number(label="Output Video Width", value=832, step=16)
                        height = gr.Number(label="Output Video Height", value=480, step=16)
                        num_input_frames = gr.Number(label="Input Frames Count fron input video", value=24, step=1)
                        num_output_frames = gr.Number(label="Output Frames Count", value=81, step=1)
                    with gr.Row():
                        num_inference_steps = gr.Number(label="Inference Steps", value=50, step=1)
                        guidance_scale = gr.Number(label="Guidance Scale", value=5.0, step=0.05)
                        seed = gr.Number(label="Seed", value=42, step=1)
                    with gr.Row():
                        teacache_treshold = gr.Number(label="TeaCache Treshold. Less coef -> Better quality, but longer inference.", interactive=True, value=0.3, precision=2, minimum=0.0, maximum=1.5, step=0.05)
                    
                    generate_button = gr.Button("🎬 Generate Video")

            with gr.Column():
                video_output = gr.Video(label="Generate Video", width=720, height=720)
                with gr.Row():
                    download_video_button = gr.File(label="📥 Download Video", visible=False)

        def generate(prompt, negative_prompt, video_input, num_inference_steps, guidance_scale, 
                     seed, width, height, num_input_frames, num_output_frames, teacache_treshold, progress=gr.Progress(track_tqdm=True)):
            previous_video = load_video(video_input)[-num_input_frames:]
            tensor = infer(
                prompt, negative_prompt, previous_video, num_inference_steps, guidance_scale, 
                seed, width, height, num_output_frames, teacache_treshold, progress=progress
            )
            video_path = save_video(tensor, fps=24)
            video_update = gr.update(visible=True, value=video_path)

            return video_path, video_update

        generate_button.click(
            generate,
            inputs=[prompt, negative_prompt, video_input, num_inference_steps, guidance_scale, 
                    seed, width, height, num_input_frames, num_output_frames, teacache_treshold, ],
            outputs=[video_output, download_video_button],
        )
    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video continuation text prompt using Wan2.2")
    parser.add_argument(
        "--base_model_path", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, help="The path of the LoRA weights to be used")
    args = parser.parse_args()
    main(args)