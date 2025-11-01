# Wan2.2 Video Continuation (Demo)
#### *The current project is still in development.
This repo contains the code for video continuation inference using [Wan2.2](https://github.com/Wan-Video/Wan2.2).  
The main idea was taken from [LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video).  

## Description
This is simple lora for Wan2.2TI transformer.
First test - rank = 64, alpha = 128.  
It was trained using around 10k video. Input video frames 16-64 and output video frames 41-81.  
Mostly attention processor has been changed for this approach. 
  
### Models  
| Model | Best input frames count | Best output frames count | Resolution |  Huggingface Link |
|-------|:-----------:|:------------------:|:------------------:|:------------------:|
| TI2V-5B  |   24-32-40   |   49-61-81   |  704x1280| [Link](https://huggingface.co/TheDenk/wan2.2-video-continuation)             |
  
  
### How to
Clone repo 
```bash
git clone https://github.com/TheDenk/wan2.2-video-continuation 
cd wan2.2-video-continuation 
```
  
Create venv  
```bash
python -m venv venv
source venv/bin/activate
```
  
Install requirements
```bash
pip install git+https://github.com/huggingface/diffusers.git
pip install -r requirements.txt
```  

### Inference examples
#### Gradio inference
```bash
python -m inference.gradio_web_demo \
    --base_model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --lora_path TheDenk/wan2.2-video-continuation
```  

  
#### Simple inference with cli
```bash
python -m inference.cli_demo \
    --video_path "resources/ship.mp4" \
    --num_input_frames 24 \
    --num_output_frames 81 \
    --prompt "Watercolor style, the wet suminagashi inks slowly spread into the shape of an island on the paper, with the edges continuously blending into delicate textural variations. A tiny paper boat floats in the direction of the water flow towards the still-wet areas, creating subtle ripples around it. Centered composition with soft natural light pouring in from the side, revealing subtle color gradations and a sense of movement." \
    --base_model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --lora_path TheDenk/wan2.2-video-continuation
```
  
  
#### Detailed Inference
```bash
python -m inference.cli_demo \
    --video_path "resources/ship.mp4" \
    --num_input_frames 24 \
    --num_output_frames 81 \
    --prompt "Watercolor style, the wet suminagashi inks slowly spread into the shape of an island on the paper, with the edges continuously blending into delicate textural variations. A tiny paper boat floats in the direction of the water flow towards the still-wet areas, creating subtle ripples around it. Centered composition with soft natural light pouring in from the side, revealing subtle color gradations and a sense of movement." \
    --base_model_path Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --lora_path TheDenk/wan2.2-video-continuation \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --video_height 480 \
    --video_width 832 \
    --negative_prompt "bad quality, low quality" \
    --seed 42 \
    --out_fps 24 \
    --output_path "result.mp4" \
    --teacache_treshold 0.5
```


#### Minimal code example
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from diffusers.utils import load_video, export_to_video
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler

from wan_continuous_transformer import WanTransformer3DModel
from wan_continuous_pipeline import WanContinuousVideoPipeline

base_model_path = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
lora_path = "TheDenk/wan2.2-video-continuation"
vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float32)
transformer = WanTransformer3DModel.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)

pipe = WanContinuousVideoPipeline.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    transformer=transformer,
    vae=vae, 
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

pipe.transformer.load_lora_adapter(
    lora_path,
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="video_continuation",
    prefix=None,
)
pipe.set_adapters("video_continuation", adapter_weights=1.0)

img_h = 480 # 704 512 480
img_w = 832 # 1280 832 768

num_input_frames = 24 # 16 24 32
num_output_frames = 81   # 81 49

video_path = 'ship.mp4'
previous_video = load_video(video_path)[-num_input_frames:]

prompt = "Watercolor style, the wet suminagashi inks slowly spread into the shape of an island on the paper, with the edges continuously blending into delicate textural variations. A tiny paper boat floats in the direction of the water flow towards the still-wet areas, creating subtle ripples around it. Centered composition with soft natural light pouring in from the side, revealing subtle color gradations and a sense of movement."
negative_prompt = "bad quality, low quality"

output = pipe(
    previous_video=previous_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=img_h,
    width=img_w,
    num_frames=num_output_frames,
    guidance_scale=5,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output_type="pil",

    teacache_treshold=0.4,
).frames[0]

export_to_video(output, "output.mp4", fps=16)
```


## Acknowledgements
Original code and models [Wan2.2](https://github.com/Wan-Video/Wan2.2).  
Video continuation approach from [LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video).  
Increase inference speed with [TeaCache](https://github.com/ali-vilab/TeaCache)  

## Citations
```
@misc{TheDenk,
    title={Wan2.2 Video Continuation},
    author={Karachev Denis},
    url={https://github.com/TheDenk/wan2.2-video-continuation},
    publisher={Github},
    year={2025}
}
```

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
