import clip
import torch
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from segment_anything import sam_model_registry, SamPredictor
BICUBIC = InterpolationMode.BICUBIC
import random
import string
import warnings
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="CLIP Model")
parser.add_argument("--clip_model", type=str, required=True, help="Path to the input image")
args = parser.parse_args()

clipmodel = args.clip_model

# Get CLIP input dims
model_to_dims = {
    'RN50': 224, 'RN101': 224, 'ViT-B/32': 224, 'ViT-B/16': 224, 'ViT-L/14': 224,
    'RN50x4': 288, 'RN50x16': 384, 'RN50x64': 448, 'ViT-L/14@336px': 336
}

input_dims = model_to_dims.get(clipmodel)

# Initialize CLIP and SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load(clipmodel, device=device)
model.eval()
preprocess = Compose([Resize((input_dims, input_dims), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                      Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

image_dir = "IMG_IN"
texts_dir = "TOK"

output_directory = "out"

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)
    print(f"Created output directory: {output_directory}")
    
def unique_id(length=5):
    """Generate a random string of letters and digits."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    
print("\nRunning CLIP Surgery to see what CLIP was looking at...\n")

# Loop through each image and text in the directory
for image_filename in os.listdir(image_dir):
    if image_filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_dir, image_filename)
        base_filename = os.path.splitext(image_filename)[0]
        csv_filename = f"tokens_{base_filename}.csv"
        csv_path = os.path.join(texts_dir, csv_filename)

        # Check if corresponding CSV exists
        if os.path.exists(csv_path):
            # Read CSV and prepare texts
            df = pd.read_csv(csv_path)
            all_texts = df['Token'].tolist()
            # Set threshold of words that occured at least X times for later steps.Default >= 3
            target_texts = df[df['Frequency'] >= 3]['Token'].tolist()

            pil_img = Image.open(image_path)
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            image = preprocess(pil_img).unsqueeze(0).to(device)

         
            with torch.no_grad():
                # Extract image features
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                # Prompt ensemble for text features with normalization
                text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)

                # Similarity map from image tokens with min-max norm and resize, B,H,W,N
                features = image_features @ text_features.t()
                similarity_map = clip.get_similarity_map(features[:, 1:, :], cv2_img.shape[:2])

                # Draw similarity map
                for b in range(similarity_map.shape[0]):
                    for n in range(similarity_map.shape[-1]):
                        if all_texts[n] not in target_texts:
                            continue
                        vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
                        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                        vis = cv2_img * 0.4 + vis * 0.6
                        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                        print('CLIP:', all_texts[n])
                        plt.title('CLIP:' + all_texts[n])
                        plt.imshow(vis)
                        random_str = unique_id()
                        unique_identifier = f"{random_str}"
                        plt.savefig(f"OUT/{base_filename}_{all_texts[n]}_CLIP-{unique_identifier}.png")
            plt.clf()
            
            model, preprocess = clip.load(f"CS-{clipmodel}", device=device)
            model.eval()

            with torch.no_grad():
                # CLIP architecture surgery acts on the image encoder
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                # Prompt ensemble for text features with normalization
                text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)

                # Apply feature surgery
                similarity = clip.clip_feature_surgery(image_features, text_features)
                similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

                # Draw similarity map
                for b in range(similarity_map.shape[0]):
                    for n in range(similarity_map.shape[-1]):
                        if all_texts[n] not in target_texts:
                            continue
                        vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
                        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                        vis = cv2_img * 0.4 + vis * 0.6
                        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                        print('CLIP Surgery:', all_texts[n])
                        plt.title('CLIP Surgery:' + all_texts[n])
                        plt.imshow(vis)
                        random_str = unique_id()
                        unique_identifier = f"{random_str}"
                        plt.savefig(f"OUT/{base_filename}_{all_texts[n]}_CLIP-Surgery-{unique_identifier}.png")
            plt.clf() 
                        
            # This preprocess for all next cases
            preprocess =  Compose([Resize((512, 512), interpolation=BICUBIC), ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
            image = preprocess(pil_img).unsqueeze(0).to(device)
                
            for target_text in target_texts:
                texts = [target_text]     
                with torch.no_grad():
                    # CLIP architecture surgery acts on the image encoder
                    image_features = model.encode_image(image)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)

                    # Prompt ensemble for text features with normalization
                    text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)

                    # Extract redundant features from an empty string
                    redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)

                    # Apply feature surgery for single text
                    similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
                    similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

                    # Draw similarity map
                    for b in range(similarity_map.shape[0]):
                        for n in range(similarity_map.shape[-1]):
                            vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
                            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                            vis = cv2_img * 0.4 + vis * 0.6
                            vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                            print('CLIP Surgery for a single text:', texts[n])
                            plt.title('CLIP Surgery for a single text:' + texts[n])
                            plt.imshow(vis)
                            random_str = unique_id()
                            unique_identifier = f"{random_str}"
                            plt.savefig(f"OUT/{base_filename}_{texts[n]}_CLIP-Surgery-1-text-{unique_identifier}.png")
                plt.clf() 
                            
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            predictor.set_image(np.array(pil_img))
            
            # Sub-loop for each target_text
            for target_text in target_texts:
                texts = [target_text]            
              
               
                with torch.no_grad():
                    # CLIP architecture surgery acts on the image encoder
                    image_features = model.encode_image(image)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)

                    # Prompt ensemble for text features with normalization
                    text_features = clip.encode_text_with_prompt_ensemble(model, texts, device)

                    # Extract redundant features from an empty string
                    redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device)

                    # CLIP feature surgery with costum redundant features
                    similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0]
    
                    # Inference SAM with points from CLIP Surgery
                    points, labels = clip.similarity_map_to_points(similarity[1:, 0], cv2_img.shape[:2], t=0.8)
                    masks, scores, logits = predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True)
                    mask = masks[np.argmax(scores)]
                    mask = mask.astype('uint8')

                    # Visualize the results
                    vis = cv2_img.copy()
                    vis[mask > 0] = vis[mask > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
                    for i, [x, y] in enumerate(points):
                        cv2.circle(vis, (x, y), 3, (0, 102, 255) if labels[i] == 1 else (255, 102, 51), 3)
                    vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                    print('SAM & CLIP Surgery for single text:', texts[0])
                    plt.title('SAM & CLIP Surgery for single text:' + texts[0])
                    plt.imshow(vis)
                    random_str = unique_id()
                    unique_identifier = f"{random_str}"
                    plt.savefig(f"OUT/{base_filename}_{texts[0]}_SAM-CLIP-Sur-1-text-{unique_identifier}.png")
