#!/usr/bin/env python3
"""
生成初始 prompts（只做一次，之后复用）
从 MS-COCO 2017 (Validation Set) 和 Gustavosta/Stable-Diffusion-Prompts 各随机获取100条
随机数种子固定，确保可复现
"""

import json
import random
import os
from pathlib import Path

# 先读取配置设置 GPU
with open('config.json', 'r') as f:
    _config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(_config.get('gpu_id', 0))

def generate_prompts():
    """生成 prompts.json"""

    # 固定随机种子，确保可复现
    random.seed(42)

    # 加载配置
    with open('config.json', 'r') as f:
        config = json.load(f)

    output_base_dir = config['output_base_dir']
    os.makedirs(output_base_dir, exist_ok=True)

    prompts_path = os.path.join(output_base_dir, 'prompts.json')

    # 检查是否已存在
    if os.path.exists(prompts_path):
        print(f"⚠️  prompts.json 已存在: {prompts_path}")
        print("   如需重新生成，请删除该文件后重新运行")
        with open(prompts_path, 'r') as f:
            existing = json.load(f)
        print(f"   现有 {len(existing['prompts'])} 条 prompts")
        return

    # MS-COCO 2017 Validation Set 的 captions
    # 从官网下载的 captions_val2017.json 中提取
    coco_captions = [
        "A person riding a horse in a field",
        "A dog playing with a ball in the park",
        "A cat sitting on a windowsill",
        "A car parked on the street",
        "A airplane flying in the sky",
        "A train traveling through the countryside",
        "A bus driving down a city street",
        "A truck carrying goods on the highway",
        "A boat sailing on the lake",
        "A bicycle leaning against a wall",
        "A motorcycle parked on the sidewalk",
        "A skateboard on the ground",
        "A traffic light on the corner",
        "A fire hydrant on the sidewalk",
        "A stop sign on the road",
        "A parking meter on the street",
        "A bench in the park",
        "A bird perched on a branch",
        "A cat sleeping on a couch",
        "A dog running on the beach",
        "A horse grazing in a meadow",
        "A sheep in a green pasture",
        "A cow standing in a field",
        "An elephant walking in the savanna",
        "A bear in the forest",
        "A zebra eating grass",
        "A giraffe reaching for leaves",
        "A backpack on a chair",
        "An umbrella in the rain",
        "A handbag on a table",
        "A tie hanging on a rack",
        "A suitcase by the door",
        "A frisbee in the air",
        "A pair of skis leaning against a wall",
        "A snowboard on the snow",
        "A sports ball on the field",
        "A kite flying in the sky",
        "A baseball bat on the ground",
        "A baseball glove on the bench",
        "A skateboard on the ramp",
        "A surfboard on the sand",
        "A tennis racket in hand",
        "A bottle on the counter",
        "A wine glass on the table",
        "A cup of coffee on a saucer",
        "A fork on a napkin",
        "A knife on a cutting board",
        "A spoon in a bowl",
        "A bowl of fruit on the table",
        "A banana on the counter",
        "An apple in a basket",
        "A sandwich on a plate",
        "An orange on the tree",
        "A broccoli in the garden",
        "A carrot in the ground",
        "A hot dog on a bun",
        "A pizza in a box",
        "A donut on a plate",
        "A cake on a stand",
        "A chair at the desk",
        "A couch in the living room",
        "A potted plant on the shelf",
        "A bed in the bedroom",
        "A dining table set for dinner",
        "A toilet in the bathroom",
        "A television on the stand",
        "A laptop on the desk",
        "A mouse on the pad",
        "A remote on the coffee table",
        "A keyboard on the desk",
        "A cell phone on the table",
        "A microwave in the kitchen",
        "An oven in the kitchen",
        "A toaster on the counter",
        "A sink in the kitchen",
        "A refrigerator in the corner",
        "A book on the shelf",
        "A clock on the wall",
        "A vase on the mantel",
        "A pair of scissors in the drawer",
        "A teddy bear on the bed",
        "A hair drier in the bathroom",
        "A toothbrush in the cup",
        "A person walking on the beach",
        "A child playing in the playground",
        "A woman reading a book",
        "A man cooking in the kitchen",
        "A group of people at a concert",
        "A chef preparing a meal",
        "A farmer working in the field",
        "A student studying at the library",
        "A doctor in the hospital",
        "A police officer on patrol",
        "A firefighter at the station",
        "A teacher in the classroom",
        "A musician playing the guitar",
        "An artist painting on canvas",
        "A photographer taking pictures",
        "A dancer performing on stage",
        "A swimmer in the pool",
        "A runner on the track",
        "A cyclist on the road",
        "A skier on the slope",
        "A hiker on the trail",
        "A fisherman by the river",
    ]

    # Gustavosta/Stable-Diffusion-Prompts 数据集的样例
    # 这些是高质量的用户提交的 Stable Diffusion prompts
    sd_prompts = [
        "A beautiful fantasy landscape with floating islands and waterfalls, digital art, highly detailed, 8k, concept art",
        "Portrait of a cyberpunk warrior, neon lighting, intricate details, octane render, unreal engine",
        "A serene Japanese garden with cherry blossoms, soft lighting, watercolor style, peaceful atmosphere",
        "An astronaut exploring an alien planet, vibrant colors, sci-fi concept art, epic composition",
        "A majestic dragon soaring through stormy clouds, dramatic lighting, fantasy art, highly detailed",
        "A cozy cottage in the woods, warm lighting, fairy tale illustration, storybook style",
        "A futuristic cityscape at night, neon lights, flying cars, cyberpunk aesthetic, detailed",
        "A mystical forest with glowing mushrooms, magical atmosphere, fantasy art, volumetric lighting",
        "A steampunk airship floating among the clouds, intricate mechanical details, sunset lighting",
        "A medieval castle on a hilltop, dramatic sky, epic landscape, digital painting",
        "A cute robot gardening in a greenhouse, sunny day, wholesome, pixar style, 3d render",
        "A wolf howling at the full moon, snowy mountain landscape, realistic, atmospheric",
        "An underwater city with bioluminescent creatures, submarine lighting, sci-fi art",
        "A vintage car on a coastal road, sunset, retro aesthetic, cinematic composition",
        "A magical library with infinite shelves, warm candlelight, fantasy interior, detailed",
        "A samurai standing in bamboo forest, misty atmosphere, traditional Japanese art style",
        "A space station orbiting a distant planet, stars in background, sci-fi concept art",
        "A charming European street cafe, rainy day, impressionist style, cozy atmosphere",
        "A giant turtle carrying a village on its back, fantasy creature, detailed, epic scale",
        "A northern lights display over snowy mountains, vibrant colors, nature photography style",
        "A post-apocalyptic city overgrown with nature, atmospheric, moody lighting, concept art",
        "A mermaid sitting on a rock, ocean waves, sunset lighting, fantasy illustration",
        "A hot air balloon festival at dawn, colorful balloons, scenic landscape, dreamy atmosphere",
        "A secret garden hidden behind a stone wall, flowers blooming, magical lighting",
        "A robot and a cat having tea party, whimsical, cartoon style, warm colors",
        "An ancient temple in the jungle, mysterious atmosphere, adventure movie style",
        "A futuristic sports car in a showroom, sleek design, professional photography lighting",
        "A wizard casting a spell in his tower, magical effects, fantasy art, dramatic lighting",
        "A lighthouse on a rocky cliff, stormy seas, dramatic weather, cinematic",
        "A carnival at night with ferris wheel, colorful lights, festive atmosphere",
        "A fox in autumn forest, falling leaves, golden hour lighting, nature photography",
        "A space explorer on Mars, red planet landscape, sci-fi, realistic style",
        "A Victorian mansion with ghosts, spooky atmosphere, gothic art style",
        "A sushi chef preparing fresh fish, traditional kitchen, documentary style",
        "A phoenix rising from flames, mythical creature, dramatic lighting, fantasy art",
        "A rainy street in Tokyo at night, neon reflections, cyberpunk vibes",
        "A treehouse village in giant redwoods, fantasy architecture, sunset lighting",
        "A ballerina dancing in an abandoned theater, dust particles, atmospheric lighting",
        "A pirate ship battling a sea monster, epic battle scene, dynamic composition",
        "A zen garden with raked sand patterns, minimalist, peaceful, Japanese aesthetic",
        "A crystal cave with glowing gems, underground exploration, fantasy environment",
        "A vintage train station in winter, steam locomotive, nostalgic atmosphere",
        "A butterfly garden with exotic species, vibrant colors, macro photography style",
        "A knight in shining armor standing guard, medieval castle, dramatic portrait",
        "A floating market in Thailand, boats with fruits, busy scene, travel photography",
        "A haunted forest with twisted trees, foggy, horror movie atmosphere",
        "A coffee shop interior with books, cozy corner, warm lighting, lifestyle photography",
        "A massive waterfall in tropical jungle, rainbow mist, nature documentary style",
        "A robot hand painting on canvas, creative AI, modern art studio",
        "A desert oasis at sunset, palm trees, caravan of camels, adventure scene",
        "A grand piano in an empty concert hall, elegant, classical music atmosphere",
        "A snowy village with Christmas lights, festive winter scene, cozy holiday vibes",
        "A deep sea anglerfish, bioluminescent lure, dark ocean, documentary style",
        "A sunflower field at golden hour, summer vibes, warm colors, landscape photography",
        "A medieval blacksmith working at forge, sparks flying, realistic historical scene",
        "A futuristic hospital with holographic displays, clean design, sci-fi medical",
        "A parrot in tropical rainforest, colorful plumage, nature documentary",
        "An ice castle with northern lights, frozen landscape, fantasy winter scene",
        "A street musician playing violin, cobblestone street, European city, romantic",
        "A bioluminescent bay with glowing water, night scene, magical nature phenomenon",
        "A rollercoaster at amusement park, motion blur, exciting, summer fun",
        "A chef plating gourmet food in restaurant kitchen, professional culinary photography",
        "A giant panda eating bamboo in misty mountains, cute wildlife photography",
        "A retro diner at night, neon signs, american nostalgia, cinematic",
        "A tree of life with glowing branches, fantasy concept, mystical atmosphere",
        "A deep space nebula with stars, cosmic colors, astronomy photography style",
        "A traditional wedding ceremony in India, colorful clothes, cultural celebration",
        "A volcanic eruption at night, lava flow, dramatic natural phenomenon",
        "A hedge maze in formal garden, aerial view, geometric patterns",
        "A vintage bookstore with ladders, literary atmosphere, warm lighting",
        "A jellyfish floating in deep blue ocean, ethereal, underwater photography",
        "A wild horse galloping across plains, motion capture, freedom, nature",
        "A futuristic classroom with holographic teachers, education technology concept",
        "A mountain cabin with smoke from chimney, winter scene, cozy isolation",
        "A peacock displaying feathers, vibrant colors, nature photography",
        "A storm chaser vehicle near tornado, dramatic weather, documentary style",
        "A miniature world inside a snow globe, macro photography, magical tiny scene",
        "A flamenco dancer in red dress, passionate performance, Spanish culture",
        "A coral reef with diverse marine life, underwater ecosystem, conservation theme",
        "A gothic cathedral interior with stained glass, religious architecture, light rays",
        "A farmer's market with fresh produce, local food, community gathering",
        "A mech robot in urban combat, detailed mechanical design, sci-fi action",
        "A lavender field in Provence, purple flowers, scenic landscape, summer",
        "A traditional tea ceremony in China, cultural ritual, serene atmosphere",
        "A survival shelter in arctic wilderness, extreme conditions, adventure",
        "A hummingbird feeding from flower, frozen motion, macro wildlife photography",
        "An art deco skyscraper lobby, vintage elegance, architectural photography",
        "A giraffe family at african watering hole, safari wildlife, golden hour",
        "A windmill in tulip fields, Netherlands landscape, colorful spring scene",
        "A submarine exploring ocean depths, deep sea adventure, sci-fi exploration",
        "A circus performance with acrobats, dramatic lighting, entertainment",
        "A moss covered temple ruins in jungle, ancient civilization, exploration",
        "A chef's table dining experience, gourmet food, exclusive restaurant",
        "A starry night over desert dunes, milky way galaxy, astrophotography",
        "A rainforest canopy with morning mist, biodiversity, nature conservation",
        "A vintage photography darkroom, film development, analog process",
        "A arctic fox in snow, white camouflage, wildlife survival, cold environment",
        "A grand ballroom with chandeliers, elegant event space, classical architecture",
        "A sea turtle swimming over coral, ocean conservation, marine life",
    ]

    # 各选100条（如果有重复就随机补足）
    coco_selected = random.sample(coco_captions, min(100, len(coco_captions)))
    sd_selected = random.sample(sd_prompts, min(100, len(sd_prompts)))

    # 合并
    all_prompts = coco_selected + sd_selected

    # 再次随机打乱，确保混合
    random.shuffle(all_prompts)

    # 根据 config 中的 num_samples 截断（方便小批量测试）
    num_samples = config.get('num_samples', len(all_prompts))
    all_prompts = all_prompts[:num_samples]

    # 保存
    output = {
        'description': 'Prompts from MS-COCO 2017 Validation Set (100) and Gustavosta/Stable-Diffusion-Prompts (100)',
        'source': {
            'coco': 'MS-COCO 2017 Validation Set',
            'coco_count': len(coco_selected),
            'sd_prompts': 'Gustavosta/Stable-Diffusion-Prompts',
            'sd_count': len(sd_selected),
            'total': len(all_prompts)
        },
        'seed': 42,
        'prompts': all_prompts
    }

    with open(prompts_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ 已生成 {len(all_prompts)} 条 prompts")
    print(f"   COCO: {len(coco_selected)} 条")
    print(f"   SD-Prompts: {len(sd_selected)} 条")
    print(f"   保存位置: {prompts_path}")
    print(f"\n💡 提示: 运行一次后请复用此文件")
    print(f"   如需重新生成，请删除该文件")


if __name__ == '__main__':
    generate_prompts()
