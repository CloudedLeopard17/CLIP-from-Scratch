import os
import cv2
import time
import math
import json 
import torch
import tiktoken
import torchvision
import inspect
import numpy as np 
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10 
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.distributed.nn as dist_nn


torch.set_float32_matmul_precision('high')

@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    debug: bool = False
    ddp = True
    #defaults for ddp 
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    
    batch_size: int = 160
    grad_accumulation_steps: int = 2
    warmup_epochs: int = 1
    epochs_count: int = 15
    warmup_ratio: float = 0.1
    init_learning_rate: float = 1e-3  # initial learning rate for lr scheduler
    weight_decay: float = .2
    betas = (0.9, 0.98)
    grad_clip: float = 1.0
    valid_interval: int = 1  
    data_root: str = "../datasets/" 
    dataset_name : str = "pixparse/cc3m-wds"
    num_workers: int = 5
    model_dir: str = 'models_clip_cc3m'
    log_dir: str = 'logs_clip_cc3m'
    model_name : str = 'clip_base_patch16_224'
    last_state: str|None =  None # if you want to load from a checkpoint, specify the file name here, e.g. 'clip_base_patch16_224_best.pt'
    
    #vit model hyperparameters
    img_size: int = 224
    patch_size: int = 16
    vit_dim: int = 768 
    vit_heads: int = 12 
    vit_encoder_layers: int = 12 
    dropout: float = 0.1

    #transformer hyperparameters
    context_length: int = 77
    transformer_width: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 12

    projection_dim: int = 512

class FFN(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self,x):
        return self.feed_forward(x)
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model%h ==0, "d_model should be divisible by h"
        
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model, bias= False)
        self.w_k = nn.Linear(d_model,d_model, bias= False)        
        self.w_v = nn.Linear(d_model, d_model, bias= False)  
        
        self.w_o = nn.Linear(d_model, d_model, bias= False)  
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        scores = query@key.transpose(-2,-1)/math.sqrt(d_k)
        #print(query.shape,key.shape, value.shape, scores.shape, mask.shape)
        if mask is not None:
            scores.masked_fill_(mask==0, -1e9)
        scores = scores.softmax(dim=-1)
        
        ### do we need ?
        if dropout is not None:
            scores = dropout(scores)
            
        return scores@value,scores 
        
    def forward(self, q, k, v, mask):
        # batch, seq len, embed dim
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # batch, h, seq, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        #x, scores = self.attention(query, key, value, mask, self.dropout)
        #using inbuilt flash attention
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.dropout.p)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)
        
        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, h:int, d_ff:int, dropout:float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_ln = self.layer_norm_1(x)
        attention = self.attention(x_ln, x_ln, x_ln, mask)
        x = x + self.dropout(attention)
        
        ffn_out = self.ffn(self.layer_norm_2(x))
        x = x+ self.dropout(ffn_out)
        
        return x
        

class Transformer(nn.Module):
    def __init__(self,d_model:int, h:int, d_ff:int, dropout:float, N=6):
        super().__init__()
        
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(d_model, h, d_ff, dropout)
                for _ in range(N)
            ]
        )
        
    
    def forward(self,x, mask=None):
        for layer in  self.encoder:
            x = layer(x, mask)
        return x
    

def img_to_patches(img, patch_size):
    bs, c, h, w = img.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by the patch size."
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    img = img.reshape(bs, c, num_patches_h, patch_size, num_patches_w, patch_size)
    img = img.permute(0, 2, 4, 1, 3, 5).contiguous()  # (batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
    patches = img.view(bs, num_patches_h * num_patches_w, c * patch_size * patch_size)  # (batch_size, num_patches, patch_dim)
    return patches

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, projection_dim=512,
                  dim =768, heads=8, encoder_layers=12, dropout=0.1):
        super(ViT, self).__init__() 
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.linear_proj = nn.Linear(3*patch_size*patch_size, dim)
        self.transformer = Transformer(dim, heads, dim*4, dropout, encoder_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.embedding = nn.Parameter(torch.randn(1, self.num_patches+1, dim))
        self.dropout = nn.Dropout(dropout)
        self.ln_pre = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)
        self.projection_layer = nn.Parameter(torch.randn(dim, projection_dim))

    def forward(self, x):
        x = img_to_patches(x, self.patch_size)  # (batch_size, num_patches, patch_dim)
        x = self.linear_proj(x)  # (batch_size, num_patches, dim)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)  # (batch_size, 1, dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, num_patches + 1, patch_dim)
        x = x + self.embedding  # (batch_size, num_patches + 1, dim)
        x = self.dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)  # (batch_size, num_patches + 1, dim)
        x = self.ln_post(x)
        x = x[:, 0]  # (batch_size, dim)
        x = torch.matmul(x, self.projection_layer)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, context_length, h,
                  d_ff, dropout, N, projection_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.randn(1, context_length, d_model))
        self.transformer = Transformer(d_model, h, d_ff, dropout, N)
        self.ln_post = nn.LayerNorm(d_model)
        self.projection_layer = nn.Parameter(torch.randn(d_model, projection_dim))

    def forward(self, x):
        masked = x.clone()
        masked[:,0]=-1
        pos = masked.argmax(dim=-1)

        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        x = x + self.positional_embedding[:, :x.shape[1], :]  # (batch_size, seq_len, d_model)
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        x = self.ln_post(x)
        x = x[torch.arange(x.shape[0]), pos, :]  # (batch_size, d_model)
        x = torch.matmul(x, self.projection_layer)  # (batch_size, projection_dim)
        return x
    
class CLIP(nn.Module):
    def __init__(self, config: TrainingConfiguration, tokenizer: tiktoken.core.Encoding):
        super().__init__()
        self.image_encoder = ViT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            projection_dim=config.projection_dim,
            dim =config.vit_dim,
            heads=config.vit_heads,
            encoder_layers=config.vit_encoder_layers,
            dropout=config.dropout
        )
        self.text_encoder = TextEncoder(
            vocab_size=tokenizer.n_vocab,
            d_model=config.transformer_width,
            context_length=config.context_length,
            h=config.transformer_heads,
            d_ff=config.transformer_width*4,
            dropout=config.dropout,
            N=config.transformer_layers,
            projection_dim=config.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        #self.initialize_weights()
    def encode_image(self, images):
        image_features = self.image_encoder(images)  # (batch_size, projection_dim)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, tokens):
        text_features = self.text_encoder(tokens)  # (batch_size, projection_dim)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, tokens):
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokens)

        return image_features, text_features


def transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def tokenize(txt, tokenizer, context_length=77):
    '''
        GPT2 don't have sos and pad tokens but we 
        can use eot token as both sos and pad token
    '''
    eos_token_id = tokenizer.eot_token
    sos_token_id = eos_token_id
    pad_token_id = eos_token_id

    tokens = tokenizer.encode(txt)
    tokens = tokens[:context_length - 2]
    tokens = [sos_token_id] + tokens + [eos_token_id]
    tokens += [pad_token_id] * (context_length - len(tokens))

    return np.array(tokens, dtype=np.int32)



class CC3MArrowDataset(Dataset):
    def __init__(self, arrow_dataset, transform, tokenizer):
        self.dataset = arrow_dataset
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        try:
            image = sample["jpg"].convert("RGB")  # already PIL, just ensure RGB
            caption = sample["txt"]

            image = self.transform(image)
            tokens = tokenize(caption, self.tokenizer)

            return (
                image,
                torch.tensor(tokens, dtype=torch.int32),
            )
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self)))
    def __len__(self):
        return len(self.dataset)


def clip_loss(image_features, text_features, logit_scale, gather=True):
    """
    gather=True  → all_gather across GPUs (use on sync steps)
    gather=False → local batch only (use on accumulation steps)
    """
    if gather and dist.is_initialized() and dist.get_world_size() > 1:
        # dist_nn.all_gather supports gradients natively — no custom autograd needed
        all_image = torch.cat(dist_nn.all_gather(image_features), dim=0)
        all_text  = torch.cat(dist_nn.all_gather(text_features),  dim=0)
    else:
        all_image = image_features
        all_text  = text_features
 
    scale        = logit_scale.exp()
    logits_image = scale * all_image @ all_text.t()   # (full_B, full_B)
    logits_text  = scale * all_text  @ all_image.t()  # (full_B, full_B)
 
    labels     = torch.arange(all_image.shape[0], device=image_features.device)
    loss_image = F.cross_entropy(logits_image, labels)
    loss_text  = F.cross_entropy(logits_text,  labels)
    loss       = (loss_image + loss_text) / 2
 
    return loss, loss_image, loss_text
 
def train(
    config: TrainingConfiguration,
    model: torch.nn.parallel.distributed.DistributedDataParallel, 
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader, 
    epoch_idx: int, 
    global_step: int,
    lr_scheduler, 
    summary_writer: SummaryWriter|None,
    device: str,
    device_type: str
) -> tuple[float, int]:

    # change model in training mood
    model.train()
    # to get batch loss
    batch_loss = np.array([])
    norms = np.array([])
    batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch_idx}") if not config.ddp else train_loader
    t0 = time.time()
    for batch_idx, (imgs, tokens) in enumerate(batch_iterator):
        
        imgs = imgs.to(device)
        tokens = tokens.to(device)
        is_sync_step = (batch_idx + 1) % config.grad_accumulation_steps == 0   

        if config.ddp:
            model.require_backward_grad_sync = ((batch_idx+1)%config.grad_accumulation_steps == 0)
        
        
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # forward pass to the model
            image_features, text_features = model(imgs, tokens)
            loss, loss_image, loss_text = clip_loss(image_features, text_features,
                                                model.module.logit_scale,\
                                                gather= is_sync_step)    

        batch_loss = np.append(batch_loss, [loss.item()])
        loss = loss / config.grad_accumulation_steps
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        norms = np.append(norms, [norm.item()])
        
        if is_sync_step:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if lr_scheduler:
                lr_scheduler.step()
            if config.master_process and summary_writer is not None:
                summary_writer.add_scalar('Learning Rate',optimizer.param_groups[0]['lr'] , global_step)
                summary_writer.add_scalar('Step/Train Loss', loss.item()*config.grad_accumulation_steps, global_step)

        if not config.ddp:
            batch_iterator.set_postfix({"loss": f"{loss.item()*config.grad_accumulation_steps:.3f}",\
                                        "grad_norm": f"{norm.item():.3f}"})

    dt = time.time() - t0        
    epoch_loss = batch_loss.mean()
    epoch_norm = norms.mean()
    if config.master_process:
        print('Epoch: {} Train Loss: {:.3f} Grad Norm: {:.3f} Time: {:.3f} sec'.format(epoch_idx, epoch_loss, epoch_norm, dt))
    return epoch_loss, global_step

def validation(
    config: TrainingConfiguration, 
    model: torch.nn.parallel.distributed.DistributedDataParallel,
    test_loader: torch.utils.data.DataLoader, 
    epoch_idx: int,
    device: str,
    device_type: str
) -> float:

    model.eval()
    # to get batch loss
    batch_loss = np.array([])
    with torch.no_grad():
        batch_iterator = tqdm(test_loader, desc=f"Validating Epoch {epoch_idx}")
        for batch_idx, (imgs, tokens) in enumerate(batch_iterator):

            imgs = imgs.to(device)
            tokens = tokens.to(device)
            is_sync_step = False    
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                image_features, text_features = model(imgs, tokens)
                loss, loss_image, loss_text = clip_loss(image_features, text_features,
                                                    model.module.logit_scale,\
                                                    gather= is_sync_step)    


            
            batch_loss = np.append(batch_loss, [loss.item()])
            batch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})

                
    epoch_loss = batch_loss.mean()
    print('Epoch: {} Valid Loss: {:.6f}'.format(epoch_idx, epoch_loss))
    return epoch_loss

def zero_shot_evaluation(config: TrainingConfiguration,
                model: torch.nn.parallel.distributed.DistributedDataParallel,
                tokenizer: tiktoken.core.Encoding,
                device: str,
                )-> float:
    
    dataset_zero_shot = CIFAR10(root = config.data_root, train=False, download=False, transform=transform())
    loader = DataLoader(dataset_zero_shot, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    prompts = ["a photo of a {}".format(cls) for cls in dataset_zero_shot.classes]
    tokenized_captions = np.array([tokenize(prompt, tokenizer) for prompt in prompts])
    tokenized_captions = torch.tensor(tokenized_captions, dtype=torch.int32).to(device)
    model.eval()
    with torch.no_grad():
        text_features = model.module.text_encoder(tokenized_captions.to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        correct = total = 0
        batch_iterator = tqdm(loader, desc=f"Zero-Shot Evaluation CIFAR10")
        for images, labels in batch_iterator:
            images = images.to(device)
            img_features = model.module.image_encoder(images)     # (B, 512)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)  # (B, 512)
            sims = img_features @ text_features.T              # (B, 10)
            preds = sims.argmax(dim=-1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    print(f"Zero-shot Accuracy: {accuracy:.4f}")

    return accuracy

def save_model(model, optimizer, epoch, device, state_file_name, state_dir, valid_loss, global_step, lr,ddp = False):
    
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)

    state_path = os.path.join(state_dir, state_file_name + '_' + str(epoch)+ '.pth')

    
    # make sure you transfer the model to cpu.
    if device == 'cpu':
        model.to('cpu')
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    # save the state_dict
    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': torch.tensor(epoch),
        'global_step': torch.tensor(global_step),
        'val_loss': torch.tensor(valid_loss),   
        'learning_rate': torch.tensor(lr)
    }, state_path)
    
    #transfer model to gpu again
    if device == 'cuda':
        model.to('cuda')
    
    return

def load_model(model, state_file_name ,state_dir=TrainingConfiguration.model_dir, ddp=False):
    state_path = os.path.join(state_dir, state_file_name)

    # loading the model and getting model parameters by using load_state_dict
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    state = torch.load(state_path)
    raw_model.load_state_dict(state['model_state_dict'])
    return model

def main(
    model: torch.nn.parallel.distributed.DistributedDataParallel,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    training_configuration: TrainingConfiguration,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokenizer: tiktoken.core.Encoding,
    device: str,
    device_type: str
    ):

    if config.master_process:
        summary_writer = SummaryWriter(log_dir=config.log_dir)
    else:
        summary_writer = None

    # DistributedSampler handles splitting across GPUs automatically
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=config.ddp_world_size,
        rank=config.ddp_rank,
        shuffle=True,
        drop_last=True
    )


    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=training_configuration.batch_size,
        sampler=sampler,
        num_workers=training_configuration.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=training_configuration.batch_size,
        num_workers=training_configuration.num_workers
    )
    
    model.to(device)

    best_acc = -torch.tensor(np.inf)
    t_begin = time.time()
    start_epoch=0
    global_step = 0
    # if config.last_state is not None:
    #     model = load_model(model, config.last_state, config.model_dir, ddp=config.ddp)
    #     state_path = os.path.join(config.model_dir, config.last_state)
    #     state = torch.load(state_path)
    #     best_loss = state['val_loss'].to(device)
    #     start_epoch = state['epoch'].item() + 1
    #     global_step = state['global_step'].item()
    #     if config.master_process:
    #         print(f"Loaded model from {config.last_state}")
    #         print(f"Best Validation Loss: {best_loss.item():.4f}, Resuming training from epoch {start_epoch}, Global step {global_step}")


    for epoch in range(start_epoch, training_configuration.epochs_count):
        
        sampler.set_epoch(epoch)  # shuffle data differently at each epoch
        train_loss, global_step = train(training_configuration, model, optimizer, train_loader,\
                            epoch, global_step, scheduler, summary_writer, device, device_type)
        
        if config.master_process:
            summary_writer.add_scalar('Epoch/Train Loss', train_loss, epoch)
            if epoch % training_configuration.valid_interval == 0:
                valid_loss = validation(training_configuration, model, valid_loader, epoch, device, device_type)
                zero_shot_acc = zero_shot_evaluation(training_configuration, model, tokenizer, device)
                if zero_shot_acc > best_acc:
                    best_acc = zero_shot_acc

                summary_writer.add_scalar('Epoch/Validation Loss', valid_loss, epoch)
                summary_writer.add_scalar('Epoch/Zero-Shot Accuracy', zero_shot_acc, epoch)
                save_model(model, optimizer, epoch, device, config.model_name, config.model_dir,\
                            valid_loss, global_step, scheduler.get_last_lr()[0], ddp=config.ddp)
                
                
        
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_begin
            speed_epoch = elapsed_time / (epoch + 1)
            eta = speed_epoch *training_configuration.epochs_count - elapsed_time

            summary_writer.add_scalar('Time Elapsed',elapsed_time, epoch)
            summary_writer.add_scalar('ETA',eta, epoch)
            
    if config.master_process:
        print("Total time: {:.2f}s Best Accuracy: {:.4f}".format(time.time() - t_begin, best_acc))

    if config.ddp:
        dist.destroy_process_group()

config = TrainingConfiguration()


if config.ddp:
    config.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if config.ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    config.ddp_rank = ddp_rank
    config.ddp_local_rank = ddp_local_rank
    config.ddp_world_size = ddp_world_size
    config.master_process = master_process
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

tokenizer = tiktoken.get_encoding('gpt2')

train_ds = load_dataset(config.dataset_name, split='train', cache_dir=config.data_root)
valid_ds = load_dataset(config.dataset_name, split='validation', cache_dir=config.data_root)

if config.debug:
    train_ds = train_ds.select(range(10000))
    valid_ds = valid_ds.select(range(2000))

train_dataset = CC3MArrowDataset(train_ds, transform(), tokenizer)
valid_dataset = CC3MArrowDataset(valid_ds, transform(), tokenizer)

if config.master_process:
    print("Dataset size. Train:{} Validation: {}".format(len(train_dataset), len(valid_dataset)))



model = CLIP(config, tokenizer)
model.to(device)
model = torch.compile(model)
summary_writer = SummaryWriter(log_dir= config.log_dir)

if config.ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if config.ddp else model # always contains the "raw" unwrapped model

def get_optimizer_params(model, lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == "cuda" 
optimizer_params = get_optimizer_params(raw_model, config.init_learning_rate, config.weight_decay)

optimizer = torch.optim.AdamW(
    optimizer_params,
    lr = config.init_learning_rate,
    betas=config.betas,
    fused=use_fused
)

def get_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        # clamp so it never goes below min_lr_ratio * base_lr
        return max(cosine, min_lr_ratio)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

steps_per_epoch = math.ceil(len(train_dataset) /(config.batch_size*config.grad_accumulation_steps*config.ddp_world_size))
total_steps = steps_per_epoch * config.epochs_count  
warmup_steps = int(config.warmup_ratio * total_steps) 

if config.master_process:
    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")

scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

main(model, train_dataset, valid_dataset, optimizer, config, scheduler, tokenizer, device, device_type)




#python -m torch.distributed.run --nproc_per_node=2   clip_ddp.py 

