import torch
import torch.nn.functional as F
from model import AMCModel

def contrastive_loss(g_music, g_text, temp=0.1):
    
    g_m = F.normalize(g_music, p=2, dim=-1)
    g_t = F.normalize(g_text, p=2, dim=-1)
    
    logits = torch.matmul(g_m, g_t.T) / temp
    labels = torch.arange(logits.shape[0]).to(logits.device)
    return F.cross_entropy(logits, labels)


def train_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMCModel(v_dim=1024, a_dim=512, t_dim=768).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
   
    v = torch.randn(8, 64, 1024).to(device) 
    a = torch.randn(8, 64, 512).to(device)  
    t = torch.randn(8, 64, 768).to(device)  
    
   
    target_noise = torch.randn(8, 64, 512).to(device)
    lambda_val = 0.5

 
    pred_x, g_music = model(v, a, t)
    
    
    g_text = t.mean(dim=1) 
    
    
    l_rec = F.mse_loss(pred_x[:, :, :512], target_noise) 
    l_align = contrastive_loss(g_music, g_text)
    
    total_loss = l_rec + lambda_val * l_align
    
 
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Loss: {total_loss.item():.4f} (Rec: {l_rec:.4f}, Align: {l_align:.4f})")

if __name__ == "__main__":
    train_step()