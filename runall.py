import os
import subprocess
import argparse

# 1: Check if the folders "VIS", "TOK", and "IMG_IN" exist. If not, create them in the current directory
directories = ["TOK", "IMG_IN"]
for dir_name in directories:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 2: Check if the "IMG_IN" folder is empty
while not os.listdir("IMG_IN"):
    input("Please put some images into the IMG_IN folder. Press enter to continue...")

parser = argparse.ArgumentParser(description="CLIP")
parser.add_argument('--image_dir', type=str, default="IMG_IN", help='The directory containing the images. Defaults to "IMG_IN"')
parser.add_argument('--clip_model', type=str, default="ViT-B/16", help="CLIP model to use, default 'ViT-B/16'. Available models: 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'")
args = parser.parse_args()

if args.image_dir is None:
    raise ValueError("You must provide a path to the image folder using the argument: --image_dir \"path/to/image/folder\"")

# Get a list of all files in the directory
image_files = os.listdir(args.image_dir)
clipmodel = args.clip_model

# Loop over each file
for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(args.image_dir, image_file)    
    try:
        # Call the gradient ascent CLIP script with the image path as an argument
        clip_command = ["python", f"clipga/runclipga.py", "--image_path", image_path, "--clip_model", clipmodel]
        result = subprocess.run(clip_command, stdout=None, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    except KeyboardInterrupt:
        stop_thread = True
        t.join()
        print("\nProcess interrupted by the user.")
        sys.exit(1)
    

    if result.returncode == 0:
        print(f"CLIP tokens saved to .csv file.")
        # Continue with processing the output tokens
    else:
        print("\nCLIP script encountered an error (below). Continuing with next image (if applicable)...\n\n")
        print("Error details:", result.stderr)

print("\n\nDONE: Check the 'TOK' folder for the text opinions!")

# Call the gradient ascent CLIP script with the image path as an argument
try:
    clip_command = ["python", f"run_surgery.py", "--clip_model", clipmodel]
    result = subprocess.run(clip_command, stdout=None, stderr=subprocess.PIPE, text=True, encoding='utf-8')
except KeyboardInterrupt:
    stop_thread = True
    t.join()
    print("\nProcess interrupted by the user.")
    sys.exit(1)
    

if result.returncode == 0:
    print(f"CLIP Surgery files saved to folder.")
    # Continue with processing the output tokens
else:
    print("\nCLIP script encountered an error (below). Continuing with next image (if applicable)...\n\n")
    print("Error details:", result.stderr)


# 4: Print the final message
print("\n\nDONE: Check the 'OUT' folder for the visual results!")