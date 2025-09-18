from pathlib import Path
from collections import defaultdict, Counter
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT_DIRS      = ["../data/train", "../data/valid"]   # pool labeled data, then split
VAL_FRACTION   = 0.20                           # 80/20 split
# MINE ARE THIS TARGET_H, TARGET_W = 224, 224                   # image size
TARGET_H, TARGET_W = 64, 64
BATCH_SIZE     = 32
SEED           = 42

# ********  Check if using GPU. ********
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA version in PyTorch:", torch.version.cuda)

# ******** Set seeds. ********
# seeds for reproducibility
# when a random number is generated, it is based on this seed; consistent across runs
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ******** Get data and labels. ********
class_names = sorted(
    {p.name for rd in ROOT_DIRS # for each root dir look for class subdirectories
     for p in Path(rd).glob("*")
     if p.is_dir() # keep only class directories
     })
class_to_idx = {c:i for i,c in enumerate(class_names)} # map class name to index

samples = []  # list of (image_path, class_idx)
for rd in ROOT_DIRS:
    for c in class_names:
        cdir = Path(rd)/c # class directory
        if not cdir.exists():
            continue
        for f in cdir.iterdir(): # for each file in class directory
                samples.append((str(f), class_to_idx[c])) # (path, label)

if len(samples) == 0: # means wrong paths or no images
    raise RuntimeError("No images found. Either the ROOT_DIRS are wrong or there are no images in them.")

# ******** Stratified split. ********

# groups indices by class so we can split each class separately and maintain distribution (class ratios don't change)
by_class = {}
for i, (_, y) in enumerate(samples): # y is class index
    if y not in by_class:
        by_class[y] = []
    by_class[y].append(i)

rng = random.Random(SEED) # private random generator, i.e., not global
train_idx, val_idx = [], [] # lists of indices for train and val sets across all classes
for y, idxs in by_class.items(): # for each class
    rng.shuffle(idxs) # randomize order of indices in this class
    n = len(idxs)
    n_val = max(1, int(round(n * VAL_FRACTION))) # at least 1 for val; take 20% of samples for val
    if n > 1 and n_val >= n:  # keep at least 1 for train.
        n_val = n - 1 # if only 1 sample, it goes to val
    val_idx += idxs[:n_val] # first n_val for val
    train_idx += idxs[n_val:] # rest for train
rng.shuffle(train_idx); rng.shuffle(val_idx) # shuffle train and val indices so they aren't ordered by class

train_samples = [samples[i] for i in train_idx]
val_samples   = [samples[i] for i in val_idx]

# ******** Transforms ********
to_tensor_01 = transforms.ToTensor()  # scales pixel values from [0,255] to [0,1]
norm_neg1_1  = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # normalises tensor values channel wise; [0,1] -> [-1,1]

train_tfms = transforms.Compose([
    # resize to target size (defined beginning of file, currently original size is set, will be experimented with)
    # bilinear interpolation calculates pixel values using a weighted average of the 4 nearest pixels
    transforms.Resize((TARGET_H, TARGET_W), interpolation=transforms.InterpolationMode.BILINEAR),
    # transforms.RandomHorizontalFlip(0.5),   # randomly flip image horizontally with 50% chance (data augmentation).
    to_tensor_01, # image to tensor, scales [0,255] to [0,1]
    norm_neg1_1, # normalises he tensor values channel wise; [0,1] -> [-1,1]
])
val_tfms = transforms.Compose([
    transforms.Resize((TARGET_H, TARGET_W), interpolation=transforms.InterpolationMode.BILINEAR), # resize to target size. right now original size is set, will be experimented with
    to_tensor_01, # image to tensor, scales [0,255] to [0,1]
    norm_neg1_1, # normalises he tensor values channel wise; [0,1] -> [-1,1]
])

# ******** Flattening images ********
# MLPs expect 1D input, so we flatten the [3,H,W] image tensors to [3*H*W]

class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as img:
            x = img.convert("RGB")
            x = self.transform(x)      # [3, H, W]
        return x, y

# reads (path, label) samples and applies transforms
train_ds = ImageDataset(train_samples, train_tfms)
val_ds   = ImageDataset(val_samples,   val_tfms)

# batch loaders, shuffling for trained
# training is shuffled to avoid model learning order of data; reduces correlation between consecutive batches and prevents model from seeing data grouped by filename/class/time
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

# ******** Class weights for loss. ********
train_counts = Counter([y for _, y in train_samples]) # count of samples per class in train set
K = len(class_to_idx) # number of classes
N = sum(train_counts.values()) # total number of training samples
# guard in case a class somehow ended with 0 in train (shouldn't with robust split)
class_weights = torch.tensor(
    [N / (K * max(1, train_counts.get(i, 0))) for i in range(K)], # weight for class i
    dtype=torch.float
)

# ---- handy bits ----
input_dim = 3 * TARGET_H * TARGET_W

print(f"classes: {class_to_idx}")
print(f"train size: {len(train_ds)}   val size: {len(val_ds)}")
print("train counts:", dict(train_counts))
print("input_dim   :", input_dim)
print("class_weights:", class_weights.tolist())
