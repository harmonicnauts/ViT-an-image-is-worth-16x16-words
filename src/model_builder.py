import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(
            self, 
            in_channels:int = 3, 
            patch_size:int = 16,
            embedding_dim:int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(
            start_dim=2,
            end_dim=3
        )

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, "Input image size must be divisible to patch size"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1)

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, 
                 embedding_dim:int = 768,
                 num_heads:int = 12, # From Table 1 (ViT-Base)
                 attn_dropout:float = 0
                 ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True)
        

    def forward(self, x):
        x = self.layer_norm(x)

        attn_output, _ = self.multihead_attn(query=x, # query embeddings
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output
    

class MultiLayerPerceptronBlock(nn.Module):
    def __init__(self, 
                 embedding_dim:int = 768,
                 mlp_size:int = 3072, # MLP Size from Table 1 for ViT-Base 
                 dropout:float = 0.1
                 ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, 
                 embedding_dim:int=768,
                 num_heads:int = 12,
                 mlp_size:int = 3072,
                 mlp_dropout:float = 0.1,
                 attn_dropout:float = 0.0):
        super().__init__()

        self.msa_block = MultiHeadedSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout)
        self.mlp_block = MultiLayerPerceptronBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )

    def forward(self, x):
        x = self.msa_block(x) + x # residual
        x = self.mlp_block(x) + x

        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size:int = 224, # Training res from Table 3
        in_channels:int=3, # Input image's color channel
        patch_size:int = 16, 
        num_transformer_layer:int = 12, # Number of transformer layer (In Table 1 for ViT-Base)
        embedding_dim: int = 768, # Hidden size D from Table 1 for Vit-Base
        mlp_size:int = 3072, # MLP Size from Table 1 for Vit-Base
        num_heads:int = 12, # Number of heads from Table 1 for ViT-Base
        attn_dropout:float = 0.0, # Dropout for attention projectoin
        mlp_dropout:float = 0.1,# Dropout for MLP Layer
        embedding_dropout:float = 0.1, # Dropout for patch and position embeddings
        num_classes:int=1000 # Default for ImageNet
    ):
        super().__init__()
        
        assert img_size % patch_size == 0, "The image size must be divisible to the patch size"

        self.num_patches = (img_size * img_size) // patch_size ** 2

        self.class_embedding = nn.Parameter(data = torch.randn(
            1,1, embedding_dim
        ), requires_grad=True)

        self.position_embedding = nn.Parameter(data = torch.randn(
            1, self.num_patches + 1, embedding_dim
        ), requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        self.transformer_encoder  = nn.Sequential(*[
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout
            ) for _ in range(num_transformer_layer)] )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(
                in_features=embedding_dim,
                out_features=num_classes
            )
        )
    
    def forward(self, x):
        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1) # -1 Means to infer the dimension

        x = self.patch_embedding(x) # Patch embedding

        x = torch.cat((class_token, x), dim=1) # Prepend the class token to the beginning of the patch embedding

        x = self.position_embedding + x # Add position embedding to patch embedding

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x
