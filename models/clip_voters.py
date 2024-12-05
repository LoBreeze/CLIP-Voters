import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

pretrained_dir = '~/.cache/clip'


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class CLIP_Voter(nn.Module):
    def __init__(self, arch, pretrained_dir=pretrained_dir, logit_scale=None, thres_type='multi', num_classes=10, ratio=0.2, global_thres=150):
        '''
        args:
            -- arch: CLIP模型的名称
            -- pretrained_dir: 预训练模型的下载路径
            -- logit_scale: logits的缩放因子
            -- thres_type: 阈值类型
            -- num_classes: 类别数
            -- global_thres: 全局阈值 常数
            # -- logit_temperature: logits的温度
            -- ratio: 图像特征的融合比例
            
            # -- device: 设备
            
        '''
        super().__init__()
        self.clip_model, preprocess = clip.load(arch, device='cpu', jit=False, download_root=pretrained_dir)
        self.visual_adapter = Adapter(self.clip_model.visual.output_dim, 4)
        
        # 冻结CLIP模型的参数
        for param in self.clip_model.parameters():
            param.requires_grad = False

        
        if logit_scale is None:    
            self.logit_scale = self.clip_model.logit_scale
        else:
            self.logit_scale = logit_scale
        
        self.dtype = self.clip_model.dtype
        self.ratio = ratio
        
        if thres_type == 'multi':
            self.rejection_threshold = nn.Parameter(torch.ones(1, num_classes))
        elif thres_type == 'one':
            self.rejection_threshold = nn.Parameter(torch.tensor(1.00))
        elif thres_type == 'const':
            self.rejection_threshold = torch.tensor(global_thres) 
        # self.temperature_scale = torch.tensor(args.logit_temperature)
        
    def forward(self, image_inputs, text_inputs):
        '''
        image_inputs: processed (batch_size, C, H, W)
        text_inputs: processed, (num_classes, D)
        
        image_features: (batch_size, 512)
        text_features: (num_classes, 512)
        '''
        image_features = self.clip_model.encode_image(image_inputs)
        text_features = self.clip_model.encode_text(text_inputs) # torch.Size([10, 512])
        
        x = self.visual_adapter(image_features)
        adapter_features = self.ratio * x + (1 - self.ratio) * image_features # torch.Size([128, 512])

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # calculate the clip logits        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        clip_logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = clip_logit_scale * image_features @ text_features.t()
        
        # calculate the voter logits
        adapter_features = adapter_features / adapter_features.norm(dim=-1, keepdim=True)
        adapter_logit_scale = self.logit_scale.exp()
        
        logits = adapter_logit_scale * adapter_features @ text_features.t() # (batch_size, num_classes) torch.Size([128, 10])

        return logits, logits_per_image

if __name__ =='__main__':
    _tokenizer = _Tokenizer()
    device = 'mps'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cifar_10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    prompts = [f'a photo of a {label}' for label in cifar_10]
    text_inputs = clip.tokenize(prompts).to(device) # (num_classes, 77)
    
    # 创建测试输入
    batch_size = 128
    image_size = 224
    image_inputs = torch.randn(batch_size, 3, image_size, image_size).to(device)

    
    # 读取图像
    preprocess = clip.load("ViT-B/32", device=device)[1]
    from PIL import Image
    image = preprocess(Image.open("/Users/utopia/Documents/blip2/voters/CLIP/CLIP.png")).unsqueeze(0).to(device) # torch.Size([1, 3, 224, 224])
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ViT-B/32')
    parser.add_argument('--pretrained_dir', type=str, default='~/.cache/clip')
    parser.add_argument('--logit_scale', type=float, default=None)
    parser.add_argument('--thres_type', type=str, default='multi')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--global_thres', type=float, default=0.5)
    parser.add_argument('--logit_temperature', type=float, default=1.0)
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='mps')  # 添加设备参数
    args = parser.parse_args()
    
    model = CLIP_Voter(args)
    model = model.to(device)
    

    
    with torch.no_grad():
        logits, logits_per_image = model(image_inputs, text_inputs)
        # logits = (logits - model.rejection_threshold) * model.temperature_scale
        logits = (logits - model.rejection_threshold)
    print(logits)
    
    
