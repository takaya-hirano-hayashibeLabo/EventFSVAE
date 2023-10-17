"""
一番シンプルな入出力関係を確認するプログラム
あとはlossを小さくすれば学習できるはず
"""

import sys
from pathlib import Path
PARENT=str(Path(__file__).parent)
ROOT=str(Path(__file__).parent.parent)
sys.path.append(ROOT)

import argparse

import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
DEVICE=torch.Tensor([1,1]).device
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as arani

from src.network_parser import parse
from src import global_v as glv
from src.event_fsvae import EventFSVAE,EventFSVAELarge

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--is_view",default=False)
    parser.add_argument('--config', action='store', dest='config', help='The path of config file')
    args=parser.parse_args()
    
    # >> 一様分布かつ丸い入力をつくってるだけ >>
    M,N,T=32,32,16
    xx,yy=np.meshgrid(np.arange(M),np.arange(N))
    input=np.array([
        [
        np.where(((x-M/2)**2+(y-N/2)**2)<(M/3)**2,
                 np.where(np.random.uniform(0,1,x.shape)>0.85,1,0)
                 ,0)
        for x,y in zip(xx,yy)
        ]
        for _ in range(T)
    ])
    # >> 一様分布かつ丸い入力をつくってるだけ >>
    
    
    #>> 初期設定 >>
    params = parse(args.config)
    network_config = params['Network']
    glv.init(network_config, [DEVICE])
    
    if network_config["model"]=="FSVAE":
        net =EventFSVAE()
    elif  network_config["model"]=="FSVAE_large":
        net=EventFSVAELarge()    
    #>> 初期設定 >>

    
    #>> networkに入力 >>
    # FSVAE : [batch x chennel x 32 x 32 x T]
    # FSVAE_large : [batch x chennel x 64 x 64 x T]
    input_tr=torch.Tensor(input).permute(1,2,0) #T次元を一番後ろに
    input_tr=input_tr.unsqueeze(0) #channel次元
    input_tr=input_tr.unsqueeze(0) #バッチ次元
    x_recon,q_z,p_z,sample_z=net(input_tr, scheduled=network_config['scheduled'])

    print(f"input_tr_dim:{input_tr.shape}")
    print(f"reconst_tr_dim:{x_recon.shape}")
    #>> networkに入力 >>
    
    
    if args.is_view:
        x_recon_np=x_recon.cpu().detach().numpy()
        fig,ax=plt.subplots(1,2)
        ims=[]
        for t in range(T):
            ims+=[
                [ax[0].imshow(input[t],cmap="gray"),
                ax[1].imshow(x_recon_np[0,0,:,:,t],cmap="gray")]
                  ]
        ani=arani(fig,artists=ims,interval=100)
        ani.save(f"{PARENT}/test_input_output/in_out_view.gif")
        fig.clear()
        plt.close()
    
    
if __name__ == '__main__':
    main()
    
    