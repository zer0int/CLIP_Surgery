
import imageio
import torchvision
import PIL.Image
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#checkin_step = training_iterations - 1
checkin_step = 10
import os
import sys
import clip
import kornia
import torch
import torch.nn.functional as F
import random
clip.available_models()
import numpy as np
import argparse
import glob
from multiprocessing import cpu_count
from ldmutil import parallel_data_prefetch
from tqdm import tqdm
from torchvision.transforms import Resize
import warnings
from prodigyopt import Prodigy
import pickle
import warnings
from colorama import Fore, Style
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties

emoji_font = FontProperties(fname='C:/Windows/Fonts/seguiemj.ttf', size=12)

print("\nRunning CLIP gradient ascent to generate some text opinions...\n")

# Global dictionary to hold token frequencies
token_frequencies = {}

training_iterations = 400    # <50 will yield awfully imprecise results, >600 doesn't improve reasonably. Recommended 100-400.
batchsize = 16

parser = argparse.ArgumentParser(description="CLIP")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
parser.add_argument("--clip_model", type=str, required=True, help="Path to the input image")
args = parser.parse_args()

clipmodel = args.clip_model

print(f"Using {clipmodel} for run.")

perceptor, preprocess = clip.load(clipmodel, jit=False)
perceptor = perceptor.eval().float()

model_to_dims = {
    'RN50': 224, 'RN101': 224, 'ViT-B/32': 224, 'ViT-B/16': 224, 'ViT-L/14': 224,
    'RN50x4': 288, 'RN50x16': 384, 'RN50x64': 448, 'ViT-L/14@336px': 336
}



input_dims = model_to_dims.get(clipmodel)

  
def prepare_data_for_csv(token_frequencies, temperature, iteration, csv_file_path='data.csv'):
    # Step 1: Flatten the token frequencies into a list of tuples (token, frequency, temperature, iteration)
    data = []
    tokens_dict = token_frequencies.get(iteration, {}).get(temperature, {})
    for token, freq in tokens_dict.items():
        data.append((token, freq, temperature, iteration))
    
    # Step 2: Convert to DataFrame
    df = pd.DataFrame(data, columns=['Token', 'Frequency', 'Temperature', 'Iteration'])
    
    # Step 3: Pivot for Heatmap
    csv_data = df.pivot(index='Token', columns='Temperature', values='Frequency')
    
    csv_path = f"TOK/tokens_{img_name}.csv"
    # Step 4: Export the original DataFrame to CSV
    # Ensure the DataFrame is in a suitable format for CSV export if necessary
    csv_file_path = df.to_csv(csv_path, index=False)
    
    # Return both the heatmap DataFrame and the path to the CSV file
    return csv_file_path
  
   
"""# Def"""

def displ(img, pre_scaled=True):
  img = np.array(img)[:,:,:]
  img = np.transpose(img, (1, 2, 0))
  if not pre_scaled:
    img = scale(img, 48*4, 32*4)
  imageio.imwrite(str(3) + '.png', np.array(img))
  return display.Image(str(3)+'.png')


"""# Internal tweaks"""

def clip_encode_text(gobble, text):
  x = torch.matmul(text, gobble.token_embedding.weight)  # [batch_size, n_ctx, d_model]

  x = x + gobble.positional_embedding
  x = x.permute(1, 0, 2)  # NLD -> LND

  x = gobble.transformer(x)
  x = x.permute(1, 0, 2)  # LND -> NLD
  x = gobble.ln_final(x)

  x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ gobble.text_projection
  #print("Text embeddings shape:", x.shape)

  return x

"""# Settings"""

import warnings
warnings.filterwarnings('ignore')

batch_size = batchsize
many_tokens = 5

# a prompt to use before the learned tokens/words
prompt = clip.tokenize('''''').numpy().tolist()[0]
print("Tokenized Prompt:", prompt)
prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]


sideX = input_dims 
sideY = input_dims

# set the image to use
img_path = args.image_path

import os
img_name = os.path.splitext(os.path.basename(img_path))[0]

im = torch.tensor(imageio.imread(img_path).copy()).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255 # 0,3,1,2 . 255
im = F.interpolate(im, (sideX, sideY))
print("Image Shape After Preprocessing:", im.shape)

"""
# Setup parameters"""

torch.cuda.empty_cache()

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        ptt = prompt

        self.prompt = torch.zeros(batch_size, len(ptt), 49408).cuda()
        for jk, pt in enumerate(ptt):
          self.prompt[:, jk, pt] = 1 
        
        self.pad = torch.zeros(batch_size, 77 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1


       
    def forward(self):
      self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
      fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
      #print("Output shape after forward pass:", fin.shape)
      return fin


lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{'params': mapper, 'lr': 5}])

eps = 0

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
).cuda()

tok = clip.simple_tokenizer.SimpleTokenizer()

bests = {1000:'None', 1001:'None', 1002:'None', 1003:'None', 1004:'None'}

torch.argmax(lats(), 2)[0].clone().detach().cpu().numpy()

"""# Train"""

def augment(into):
  into = augs(into)
  return into


def ascend_txt():
  global im
  iii = nom(augment(im[:,:3,:,:].expand(64, -1, -1, -1)))
  iii = perceptor.encode_image(iii).detach()
  lll = lats()
  tx = clip_encode_text(perceptor, lll)
  return -100*torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, batch_size).T.mean(1), lll

def train():
  loss1, lll = ascend_txt()
  loss = loss1.mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss1, lll
  
accumulated_tokens = set()

def checkin(loss, lll, iteration, temperature):
    unique_tokens = set()
    global token_frequencies
    global accumulated_tokens
    
    # Initialize sub-dictionary for the current iteration and temperature if not present
    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
        decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
        decoded_tokens = decoded_tokens.replace('.', '').replace(',', '').replace('*', '').replace('$', '').replace(')', '').replace('(', '').replace('{', '').replace('}', '').replace('&', '')
        decoded_tokens = decoded_tokens.replace('#', '').replace('%', '').replace('^', '').replace('\\', '').replace('/', '').replace('/', '').replace('-', '').replace('_', '').replace('?', '')
        decoded_tokens = decoded_tokens.replace('!', '').replace('>', '').replace('<', '').replace('"', '').replace("'", '').replace(':', '').replace(';', '').replace('@', '')
        decoded_tokens = decoded_tokens.replace('=', '').replace('|', '').replace('+', '')
        cleaned_tokens = ''.join(c for c in decoded_tokens if c.isprintable()).split()
        
        if iteration not in token_frequencies:
            token_frequencies[iteration] = {}
        if temperature not in token_frequencies[iteration]:
            token_frequencies[iteration][temperature] = {}

        for token in cleaned_tokens:
            accumulated_tokens.add(token)
            if token not in token_frequencies[iteration][temperature]:
                token_frequencies[iteration][temperature][token] = 1
            else:
                token_frequencies[iteration][temperature][token] += 1
        
        #And now, save them:
        csv_file_path = prepare_data_for_csv(token_frequencies, temperature, iteration, kj)

        if loss[kj] < sorted(list(bests.keys()))[-1]:
            decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
            decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
            decoded_tokens = decoded_tokens.replace('.', '').replace(',', '').replace('*', '').replace('$', '').replace(')', '').replace('(', '').replace('{', '').replace('}', '').replace('&', '')
            decoded_tokens = decoded_tokens.replace('#', '').replace('%', '').replace('^', '').replace('\\', '').replace('/', '').replace('/', '').replace('-', '').replace('_', '').replace('?', '')
            decoded_tokens = decoded_tokens.replace('!', '').replace('>', '').replace('<', '').replace('"', '').replace("'", '').replace(':', '').replace(';', '').replace('@', '')
            decoded_tokens = decoded_tokens.replace('=', '').replace('|', '').replace('+', '')
            cleaned_tokens = ''.join(c for c in decoded_tokens if c.isprintable()).split()

            # Update token frequencies
            for token in cleaned_tokens:
                if token not in token_frequencies[iteration][temperature]:
                    token_frequencies[iteration][temperature][token] = 1
                else:
                    token_frequencies[iteration][temperature][token] += 1

            # Now, token frequencies for this sample
            # sec_csv_file_path = prepare_sec_data_for_csv(token_frequencies, temperature, iteration)
            
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                #decoded_tokens = decoded_tokens.replace('end_of_text', '').replace('start_of_text', '')
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                #print(f"Sample {kj} Tokens: {decoded_tokens}")
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}")
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue


def loop():
    global accumulated_tokens
    for i in range(training_iterations):
        loss, lll = train()
        if i % checkin_step == 0:
            checkin(loss, lll, iteration=i, temperature=1000)
    # After the loop is done, write the accumulated tokens to a file
    with open(f"TOK/accumulated_tokens_{img_name}.txt", "a", encoding='utf-8') as f:
        f.write(" ".join(accumulated_tokens))
      
loop()