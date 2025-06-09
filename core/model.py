# core/model.py
import torch # Bu satırı ekleyin
import torch.nn as nn
from torchvision import transforms
import timm

class DualInputViT(nn.Module):
    def __init__(self, num_classes=90, pretrained=False):
        super(DualInputViT, self).__init__()
        self.vit_original = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)
        self.vit_edge = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)
        embed_dim = self.vit_original.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x_original, x_edge):
        feat_original = self.vit_original(x_original)
        feat_edge = self.vit_edge(x_edge)
        combined_features = torch.cat((feat_original, feat_edge), dim=1)
        output = self.classifier(combined_features)
        return output

def load_multizoo_model(model_path, device):
    """
    Belirtilen yoldan MultiZoo modelini yükler.
    Dönüş: model, class_names, transforms
    """
    checkpoint = torch.load(model_path, map_location=device)

    num_classes = len(checkpoint.get('classes', []))
    if num_classes == 0:
        if 'class_to_idx' in checkpoint:
            num_classes = len(checkpoint['class_to_idx'])
            class_names = [None] * num_classes
            for name, idx in checkpoint['class_to_idx'].items():
                class_names[idx] = name
        else:
            raise ValueError("Model dosyasında sınıf bilgisi bulunamadı.")
    else:
        class_names = checkpoint['classes']
    
    model = DualInputViT(num_classes=num_classes)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    edge_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, class_names, transform, edge_transform, checkpoint.get('val_acc', 0.0), checkpoint.get('val_f1', 0.0)