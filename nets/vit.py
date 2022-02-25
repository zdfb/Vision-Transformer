import torch
import torch.nn as nn
from utils.utils import to_2tuple, _conv_filter


###### 定义ViT主体架构 ######


# 定义patchembedding层
class PatchEmbed(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size  # 输入图像尺寸
        self.patch_size = patch_size  # patch的尺寸
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # patch的数量
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)
    def forward(self, x):
        x = self.proj(x)  # (B, 3, 224, 224) -> (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 768, 14, 14) -> (B, 196, 768)
        return x

# 定义自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 多头注意力机制的头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = head_dim ** -0.5  # 归一化参数

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)  # 产生qkv
        self.attn_drop = nn.Dropout(attn_drop)  # attention_score的dropout
        self.proj = nn.Linear(dim, dim)  # 多头注意力合并之后的语义空间转化
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout
    
    def forward(self, x):
        B, N, C = x.shape  # bach_size的大小，sequence的长度， 每个token的维度

        # (B, N, C) -> (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 单独取出q, k, v
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)  # 获取归一化后的attention_score
        attn = self.attn_drop(attn)
        
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 定义MLP结构
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, drop = 0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层全连接层
        self.act = nn.GELU()  # 激活函数
        self.drop1 = nn.Dropout(drop_probs[0])  # 随机dropout
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层全连接层
        self.drop2 = nn.Dropout(drop_probs[1])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# 定义ViT的Block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4., qkv_bias = False, drop = 0., attn_drop = 0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)  # 对输入进行layernorm处理
        self.attn = Attention(dim, num_heads = num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)
        self.norm2 = nn.LayerNorm(dim)  # 对self-attention之后的结果进行layernorm处理
        mlp_hidden_dim = int(dim * mlp_ratio)  # feedforward网络中间层维度
        self.mlp = MLP(in_features = dim, hidden_features = mlp_hidden_dim, drop = drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # 残差结构
        x = x + self.mlp(self.norm2(x))
        return x


# 定义visiontransfomer架构
class VisionTransformer(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, num_classes = 100, embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4, qkv_bias = False, mlp_head = False, drop_rate = 0., attn_drop_rate = 0.):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes  # 输出类别数
        self.num_features = self.embed_dim = embed_dim  # 每个token的维度数
        
        # patch_embedding层
        self.patch_embed = PatchEmbed(img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim)
        num_patches = self.patch_embed.num_patches

        # 定义位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # 定义cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 定义patchembedding的dropout
        self.pos_drop = nn.Dropout(drop_rate)
        
        # 定义多个block
        self.blocks = nn.Sequential(*[
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, 
            drop = drop_rate, attn_drop = attn_drop_rate)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(self.num_features, num_classes)
    
    def forward(self, x):
        B = x.shape[0]  # batch_size数量

        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim =1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]

        x = self.head(x)
        return x


def vit_base_patch16_384(model_path):
    model = VisionTransformer(img_size = 384, patch_size = 16, embed_dim = 768, 
            depth = 12, num_heads = 12, mlp_ratio = 4, qkv_bias = True, num_classes = 1000)
    checkpoint = torch.load(model_path, map_location = 'cpu')
    checkpoint = _conv_filter(checkpoint)
    model.load_state_dict(checkpoint, strict = True)
    return model
