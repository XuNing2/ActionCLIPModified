from models.two_dim_sequencer import Sequencer2D
from timm import create_model
if __name__=="__main__":
    model=Sequencer2D(in_chans=8)
    import torch
    x = torch.randn(32,8,512)
    x=x.permute(1,0,2)
    x = model(x)