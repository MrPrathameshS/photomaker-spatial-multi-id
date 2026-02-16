#!/usr/bin/env python3
"""
PhotoMaker V2 CLI - Generate images without Gradio
Just edit the configuration below and run: python photomaker_cli.py
"""
print("🚀🚀🚀 RUNNING PHOTOMAKER_CLI.PY 🚀🚀🚀")

import torch
import torchvision.transforms.functional as TF
import numpy as np
import random
import os
import sys
from pathlib import Path

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter
from huggingface_hub import hf_hub_download

from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker import FaceAnalysis2, analyze_faces

from style_template import styles
from aspect_ratio_template import aspect_ratios
from insightface.utils import face_align
from PIL import Image
import PIL.Image



# ============================================================
# CONFIGURATION - Edit these values directly
# ============================================================

# Input image(s) - provide path(s) to face image(s)
INPUT_IMAGES = [
    "/teamspace/studios/this_studio/PhotoMaker/Data/Input/man_woman.jpg"  
]

# Prompt - must include 'img' trigger word
PROMPT = "a photo of man img1  and woman img2  "

# Output settings
OUTPUT_DIR = "/teamspace/studios/this_studio/PhotoMaker/Data/Output"
NUM_OUTPUTS = 2

# Style (check style_template.py for options)
STYLE_NAME = "Photographic (Default)"

# Negative prompt
NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# Output dimensions
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

# Generation parameters
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
STYLE_STRENGTH_RATIO = 20
SEED = None  # Set to None for random seed, or specify a number

# Sketch/Doodle settings (optional)
USE_SKETCH = False
SKETCH_IMAGE_PATH = None  # e.g., "./sketch.png"
ADAPTER_CONDITIONING_SCALE = 0.7
ADAPTER_CONDITIONING_FACTOR = 0.8

# ============================================================
# END OF CONFIGURATION
# ============================================================

MAX_SEED = np.iinfo(np.int32).max


def get_device():
    try:
        if torch.cuda.is_available():
            return "cuda"
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except:
        return "cpu"


def apply_style(style_name, positive, negative=""):
    default_style = "Photographic (Default)"
    p, n = styles.get(style_name, styles[default_style])
    return p.replace("{prompt}", positive), n + ' ' + negative

from diffusers.models.attention_processor import LoRAAttnProcessor2_0

from diffusers.models.attention_processor import AttnProcessor2_0




def load_pipeline(device):
    print("Loading pipeline...")
    
    base_model_path = 'SG161222/RealVisXL_V4.0'
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        torch_dtype = torch.float16
    
    print("Loading T2I adapter...")
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-sketch-sdxl-1.0", 
        torch_dtype=torch_dtype, 
        variant="fp16"
    ).to(device)
    
    print("Loading main pipeline...")
    pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
        base_model_path, 
        adapter=adapter, 
        torch_dtype=torch_dtype,
        use_safetensors=True, 
        variant="fp16",
    ).to(device)
    
    print("Loading PhotoMaker adapter...")
    photomaker_ckpt = hf_hub_download(
        repo_id="TencentARC/PhotoMaker-V2", 
        filename="photomaker-v2.bin", 
        repo_type="model"
    )
    
    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_words=["img1", "img2"],   # ✅ multi triggers
        pm_version="v2",
    )
    pipe.id_encoder.to(device)
    
    print("\n===== ATTENTION PROCESSOR AUDIT =====")
    for name, proc in pipe.unet.attn_processors.items():
        print(name, "->", type(proc).__name__)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    pipe.to(device)
    # 🔎 DEBUG HERE
    print("🔍 feature_extractor:", pipe.feature_extractor)
    print("🔍 image_processor:", pipe.image_processor)

    print("Pipeline loaded successfully!")

    return pipe


def load_face_detector(device):
    print("Loading face detector...")
    providers = ['CUDAExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    face_detector = FaceAnalysis2(
        providers=providers, 
        allowed_modules=['detection', 'recognition']
    )
    face_detector.prepare(ctx_id=0, det_size=(640, 640))
    return face_detector


def generate_image(pipe, face_detector, device):
    # Handle sketch input
    sketch_image = None
    adapter_scale = 0.
    adapter_factor = 0.
    
    if USE_SKETCH and SKETCH_IMAGE_PATH:
        from PIL import Image
        sketch_image = Image.open(SKETCH_IMAGE_PATH).convert("RGBA")
        r, g, b, a = sketch_image.split()
        sketch_image = a.convert("RGB")
        sketch_image = TF.to_tensor(sketch_image) > 0.5
        sketch_image = TF.to_pil_image(sketch_image.to(torch.float32))
        adapter_scale = ADAPTER_CONDITIONING_SCALE
        adapter_factor = ADAPTER_CONDITIONING_FACTOR

   



    output_w = OUTPUT_WIDTH or 1024
    output_h = OUTPUT_HEIGHT or 1024
    
    print(f"[Info] Output dimensions: {output_w} x {output_h}")

    # Apply style
    prompt, negative_prompt = apply_style(STYLE_NAME, PROMPT, NEGATIVE_PROMPT)

    # --------------------------------------------------
    # Load input images
    # --------------------------------------------------
    if not INPUT_IMAGES:
        raise ValueError("No input images! Edit INPUT_IMAGES in the script.")

    input_id_images = []
    for img_path in INPUT_IMAGES:
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
        input_id_images.append(load_image(img_path))

    
    # --------------------------------------------------
    # Load input images (original images)
    # --------------------------------------------------
    original_images = []

    for img_path in INPUT_IMAGES:
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
        original_images.append(load_image(img_path))

    print("🔥 Loaded original images:", len(original_images))


    # --------------------------------------------------
    # Extract face embeddings (TRUE MULTI-FACE SUPPORT)
    # --------------------------------------------------
    id_embed_list = []
    id_pixel_list = []
    input_id_images = []  # this will now be PER-FACE images
    bbox_list = []   # ✅ STORE NORMALIZED BBOXES HERE


    for img_idx, img in enumerate(original_images):

        # Convert PIL → numpy BGR for InsightFace
        img_array = np.array(img)[:, :, ::-1]

        faces = analyze_faces(face_detector, img_array)

        if len(faces) == 0:
            print(f"⚠️ No face detected in image {img_idx}")
            continue

        print(f"\n🔥 Detected {len(faces)} face(s) in image {img_idx}")

        # --------------------------------------------------
        # Extract each face as a separate identity
        # --------------------------------------------------
        # ✅ PUT DEBUG HERE
        print("Original image size:", img.width, img.height)
        for face_idx, face in enumerate(faces):

            print(f"\n🔥 Processing face {face_idx} in image {img_idx}")
            print("Raw bbox:", face.bbox)

            # --------------------------------------------------
            # 1️⃣ Normalize bounding box
            # --------------------------------------------------
            x1, y1, x2, y2 = face.bbox

            W = img.width
            H = img.height

            x1 /= W
            x2 /= W
            y1 /= H
            y2 /= H

            normalized_bbox = [x1, y1, x2, y2]
            bbox_list.append(normalized_bbox)

            print("Normalized bbox:", normalized_bbox)

           
            # --------------------------------------------------
            # 2️⃣ Extract 512D embedding
            # --------------------------------------------------
            raw_embedding = face["embedding"]

            print("   🔍 Embedding shape:", raw_embedding.shape)
            print("   🔍 Norm:", np.linalg.norm(raw_embedding))

            id_embed_list.append(torch.from_numpy(raw_embedding).float())



            
            # --------------------------------------------------
            # 3️⃣ Align face to 224x224 using landmarks
            # --------------------------------------------------
            aligned_face = face_align.norm_crop(
                img_array,
                landmark=face.kps,
                image_size=224
            )

            # Convert BGR → RGB
            aligned_face = aligned_face[:, :, ::-1]

            # Convert to PIL
            aligned_face = PIL.Image.fromarray(aligned_face)

            # --------------------------------------------------
            # 4️⃣ Preprocess aligned face for SDXL
            # --------------------------------------------------
            pixel_tensor = pipe.image_processor.preprocess(aligned_face)[0]

            print("   🔍 Pixel tensor shape:", pixel_tensor.shape)

            id_pixel_list.append(pixel_tensor)
            input_id_images.append(aligned_face)

    # --------------------------------------------------
    # Stack identities
    # --------------------------------------------------
    if len(id_embed_list) == 0:
        raise ValueError("❌ No valid faces found in any input images.")

    id_embeds = torch.stack(id_embed_list)          # [N, 512]
    id_pixel_values = torch.stack(id_pixel_list)   # [N, 3, 224, 224]
    identity_bboxes = torch.tensor(bbox_list).float()  # [N, 4]

    print("\n🔥 Total identities:", len(id_embed_list))
    print("🔥 input_id_images length:", len(input_id_images))
    print("🔥 Final id_embeds shape:", id_embeds.shape)
    print("🔥 Final id_pixel_values shape:", id_pixel_values.shape)
    print("🔥 identity_bboxes shape:", identity_bboxes.shape)

    # --------------------------------------------------
    # Multi-trigger validation (AFTER identity detection)
    # --------------------------------------------------

    num_identities = identity_bboxes.shape[0]
    active_triggers = pipe.trigger_words[:num_identities]

    trigger_token_ids = [
        pipe.tokenizer.convert_tokens_to_ids(t)
        for t in active_triggers
    ]

    input_ids = pipe.tokenizer.encode(prompt)

    print("🔍 Trigger words:", active_triggers)

    for trigger_word, trigger_id in zip(active_triggers, trigger_token_ids):

        count = input_ids.count(trigger_id)

        print(f"🔍 '{trigger_word}' appears {count} time(s)")

        if count == 0:
            raise ValueError(
                f"Trigger word '{trigger_word}' not found in prompt."
            )

        if count > 1:
            raise ValueError(
                f"Trigger word '{trigger_word}' appears multiple times."
            )

    print("✅ All trigger words validated correctly.")




    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------
    if len(id_embed_list) == 0:
        raise ValueError("No face detected! Use images with clear faces.")


    # --------------------------------------------------
    # --------------------------------------------------
    # Stack identities
    # --------------------------------------------------
    id_embeds = torch.stack(id_embed_list)          # [N, 512]
    id_pixel_values = torch.stack(id_pixel_list)   # [N, 3, H, W]

    print("\n🔥 Final stacked id_embeds shape:", id_embeds.shape)
    print("🔥 Final stacked id_pixel_values shape:", id_pixel_values.shape)

    # --------------------------------------------------
    # Add batch dimension (VERY IMPORTANT)
    # --------------------------------------------------
    id_embeds = id_embeds.unsqueeze(0)              # [1, N, 512]
    id_pixel_values = id_pixel_values.unsqueeze(0)  # [1, N, 3, H, W]

    print("🔥 After unsqueeze id_embeds:", id_embeds.shape)
    print("🔥 After unsqueeze id_pixel_values:", id_pixel_values.shape)



    # --------------------------------------------------
    # Validate alignment
    # --------------------------------------------------
    assert id_embeds.shape[0] == id_pixel_values.shape[0], \
        "Mismatch between identity embeddings and pixel tensors!"

    print("✅ Identity count:", id_embeds.shape[0])
    print("✅ Embedding dimension:", id_embeds.shape[1])


    # --------------------------------------------------
    # Handle seed
    # --------------------------------------------------
    seed = SEED if SEED is not None else random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)


    print("Starting inference...")
    print(f"[Info] Seed: {seed}")
    print(f"[Info] Prompt: {prompt}")
    
    start_merge_step = int(float(STYLE_STRENGTH_RATIO) / 100 * NUM_STEPS)
    if start_merge_step > 30:
        start_merge_step = 30
    

    # --------------------------------------------------
    # 🔥 BUILD SPATIAL MASKS
    # --------------------------------------------------

    num_identities = identity_bboxes.shape[0]
    H_mask = 128
    W_mask = 128

    base_masks = torch.zeros((num_identities, H_mask, W_mask))

    for i, bbox in enumerate(identity_bboxes):
        x1, y1, x2, y2 = bbox

        x1_i = int(x1 * W_mask)
        x2_i = int(x2 * W_mask)
        y1_i = int(y1 * H_mask)
        y2_i = int(y2 * H_mask)

        base_masks[i, y1_i:y2_i, x1_i:x2_i] = 1.0

    print("🔥 Built base_masks:", base_masks.shape)


    # --------------------------------------------------
    # 🔥 GET IDENTITY TOKEN INDICES FROM CLASS MASK
    # --------------------------------------------------

    # class_tokens_mask was created inside pipeline
    # so we extract indices dynamically during forward
    # but here we need the grouped structure

    # For multi-trigger, tokens are consecutive per identity
    # Example: [7,8, 11,12] → [[7,8], [11,12]]

    identity_token_indices = []

    # infer from trigger count and num_tokens
    num_tokens_per_id = pipe.id_encoder.num_tokens
    mask_indices = None

    # We will temporarily encode prompt to get mask
    _, _, _, _, class_tokens_mask = pipe.encode_prompt_with_trigger_word(
        prompt=prompt,
        device=device,
        num_id_images=num_identities
    )

    mask_indices = class_tokens_mask[0].nonzero().flatten().tolist()

    print("🔥 Mask indices:", mask_indices)

    for i in range(num_identities):
        start = i * num_tokens_per_id
        end = start + num_tokens_per_id
        identity_token_indices.append(mask_indices[start:end])

    print("🔥 identity_token_indices:", identity_token_indices)


    # --------------------------------------------------
    # 🔥 APPLY SPATIAL PROCESSOR TO CROSS-ATTENTION ONLY
    # --------------------------------------------------

    from photomaker.spatial_identity_attn import SpatialIdentityAttnProcessor


    new_processors = {}

    for name, proc in pipe.unet.attn_processors.items():

        if "attn2" in name:  # cross-attention only
            new_processors[name] = SpatialIdentityAttnProcessor(
                identity_token_indices,
                base_masks
            )
        else:
            new_processors[name] = proc

    pipe.unet.set_attn_processor(new_processors)

    print("✅ SpatialIdentityAttnProcessor applied to cross-attn layers only")


    print("🔥 input_id_images length:", len(input_id_images))
    pipe.num_identities = num_identities

    images = pipe(
        prompt=prompt,
        width=output_w,
        height=output_h,
        negative_prompt=negative_prompt,
        num_images_per_prompt=NUM_OUTPUTS,
        num_inference_steps=NUM_STEPS,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=GUIDANCE_SCALE,

        # ✅ Pass raw PIL images
        input_id_images=input_id_images,

        # ✅ Pass stacked identity embeddings
        id_embeds=id_embeds,

        image=sketch_image,
        adapter_conditioning_scale=adapter_scale,
        adapter_conditioning_factor=adapter_factor,
        identity_bboxes=identity_bboxes,  # 🔥 pass here
    ).images

    
    return images, seed


def main():
    print("=" * 50)
    print("PhotoMaker V2 CLI")
    print("=" * 50)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    pipe = load_pipeline(device)
    face_detector = load_face_detector(device)
    
    try:
        images, used_seed = generate_image(pipe, face_detector, device)
        
        print(f"\nSaving {len(images)} image(s) to {output_dir}/")
        for i, img in enumerate(images):
            filename = f"output_seed{used_seed}_{i+1}.png"
            filepath = output_dir / filename
            img.save(filepath)
            print(f"  Saved: {filepath}")
        
        print(f"\nDone! Generated {len(images)} image(s) with seed {used_seed}")
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
