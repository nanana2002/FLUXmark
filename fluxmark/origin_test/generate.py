import torch
from diffusers import FluxPipeline
import os

os.environ['HF_HOME'] = '/data/daiyina/hf_cache'

print("正在加载模型...")
pipe = FluxPipeline.from_pretrained(
    '/data/daiyina/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
print("模型加载完成！")

prompt = "a cat holding a sign that says hello world"
print(f"正在生成: {prompt}")

image = pipe(
    prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512,
    width=512
).images[0]

output_path = '/data/daiyina/project_flux/test_output.png'
image.save(output_path)
print(f"图片已保存到: {output_path}")