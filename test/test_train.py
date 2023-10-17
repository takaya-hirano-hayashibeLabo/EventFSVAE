"""
一番シンプルな入出力関係を学習するプログラム
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
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as arani
import pandas as pd

from src.network_parser import parse
from src import global_v as glv
from src.event_fsvae import EventFSVAE,EventFSVAELarge

def train(network:EventFSVAE,trainloader:DataLoader,opti:torch.optim.Adam,epoch):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']
    
    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0
    
    loss_table=[]
    loss_column=["epoch","batch_idx","loss","reconst","distance"]
    
    network = network.train()
    
    for batch_idx, spike_input in enumerate(trainloader):   
        opti.zero_grad()

        x_recon, q_z, p_z, sampled_z = network(spike_input, scheduled=glv.network_config['scheduled']) # sampled_z(B,C,1,1,T)
        
        if glv.network_config['loss_func'] == 'mmd':
            losses = network.loss_function_mmd(spike_input, x_recon, q_z, p_z)
        elif glv.network_config['loss_func'] == 'kld':
            losses = network.loss_function_kld(spike_input, x_recon, q_z, p_z)
            
        #>> eventだとこれが基本 >>
        elif glv.network_config["loss_func"]=="van_rossum":
            losses = network.loss_function_kld(spike_input, x_recon, q_z, p_z)
        #>> eventだとこれが基本 >>
    
        else:
            raise ValueError('unrecognized loss function')
        
        losses['loss'].backward()
        
        opti.step()
        network.weight_clipper()

        loss_table_idx=[epoch,batch_idx]+[loss.detach().cpu().item() for loss in losses.values()]
        loss_table=loss_table+[loss_table_idx]
        # print(loss_table)
        loss_pd=pd.DataFrame(loss_table,columns=loss_column)

        mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
        mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
        mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)

        print(f'Epoch[{epoch}/{max_epoch}] Batch[{batch_idx}/{len(trainloader)}] Loss: {loss_pd["loss"].mean()}, RECONS: {loss_pd["reconst"].mean()}, DISTANCE: {loss_pd["distance"].mean()}')

    mean_q_z = mean_q_z.permute(1,0,2) # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # (k,C,T)

    return loss_pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--is_train",default=False,type=bool)
    parser.add_argument('--config', action='store', dest='config', help='The path of config file')
    args=parser.parse_args()
    
    
    #>> 初期設定 >>
    params = parse(args.config)
    network_config = params['Network']
    glv.init(network_config, [DEVICE])
    
    if network_config["model"]=="FSVAE":
        net =EventFSVAE()
        net_pre_trained=EventFSVAE()

    elif  network_config["model"]=="FSVAE_large":
        net=EventFSVAELarge()    
        net_pre_trained=EventFSVAELarge()
    #>> 初期設定 >>
    
    
    # >> 一様分布かつ丸い入力をつくってるだけ >>
    class_size=10 #10パターンくらい作ってみる
    M,N,T=32,32,16
    xx,yy=np.meshgrid(np.arange(M),np.arange(N))
    input=np.array([
        [[
        np.where(((x-M/2)**2+(y-N/2)**2)<(M/3)**2,
                 np.where(np.random.uniform(0,1,x.shape)>0.85,1,0)
                 ,0)
        for x,y in zip(xx,yy)
        ]
        for _ in range(T)]
        for _ in range(class_size)])
    # >> 一様分布かつ丸い入力をつくってるだけ >>
    
    
    #>> optimizer >>
    optimizer = torch.optim.Adam(net.parameters(), 
                                lr=glv.network_config['lr'], 
                                betas=(0.9, 0.999), 
                                weight_decay=0.001)
    optimizer.param_groups[0]["capturable"]=True
    #>> optimizer >>
    
    
    #>> dataloader >>
    repeat_frac=20 #何倍のデータを複製するか
    input_tr=torch.Tensor(input).permute(0,2,3,1) #T次元を一番後ろに
    input_tr=input_tr.unsqueeze(1) #channel次元
    input_tr=input_tr.repeat(repeat_frac,1,1,1,1) #バッチ次元を複製
    dataloader=DataLoader(
        dataset=torch.Tensor(input_tr), batch_size=network_config["batch_size"],
        shuffle=True,drop_last=True,
        generator=torch.Generator(device=DEVICE),
        )
    #>> dataloader >>
    
    
    #>> 学習 >>
    # FSVAE : [batch x chennel x 32 x 32 x T]
    # FSVAE_large : [batch x chennel x 64 x 64 x T]
    result_path=f"{PARENT}/test_train"
    loss=pd.DataFrame([])
    if  args.is_train:
        for e in range(glv.network_config['epochs']):
            
            # >> posteriorの事前分布を更新してる？ >>
            if network_config['scheduled']:
                net.update_p(e, glv.network_config['epochs'])
            # << posteriorの事前分布を更新してる？ <<
                
            train_loss_ep:pd.DataFrame = train(net, dataloader, optimizer, e)
            if len(loss)==0:
                loss=train_loss_ep
            else:
                loss=pd.concat([loss, train_loss_ep],axis=0)
            loss.to_csv(f"{result_path}/fsvae_test_loss.csv",index=False)
            
        torch.save(net.to(DEVICE).state_dict(), f'{result_path}/test_train.pth')
    #>> 学習 >>
    
    
    #>> 再構成 >>
    net.eval()
    with torch.no_grad():
        net.load_state_dict(torch.load(f=f"{result_path}/test_train.pth",map_location=DEVICE))
        spike_input = input_tr[0].unsqueeze(0) # (N,C,H,W,T)
        x_recon, q_z, p_z, sampled_z = net(spike_input, scheduled=network_config['scheduled'])
        x_recon_np=(x_recon.detach().to("cpu").numpy())
        
        x_pre_trained,_,_,pre_trained_sample_z=net_pre_trained(spike_input, scheduled=network_config['scheduled'])
        x_pre_trained_np=x_pre_trained.detach().to("cpu").numpy()
    # print(sampled_z.shape)
    #>> 再構成 >>
        
        
    #>> 描画 >>
    fig,ax=plt.subplots(1,3)
    ax[0].set_title("input spike")
    ax[1].set_title("pre-trained\noutput spike")
    ax[2].set_title("trained\noptput spike")
    ims=[]
    for t in range(T):
        ims+=[
            [ax[0].imshow(input[0][t],cmap="gray"),
            ax[1].imshow(x_pre_trained_np[0,0,:,:,t],cmap="gray"),
            ax[2].imshow(x_recon_np[0,0,:,:,t],cmap="gray")]
                ]
    ani=arani(fig,artists=ims,interval=100)
    ani.save(f"{PARENT}/test_train/in_trained_view.gif")
    fig.clear()
    plt.close()
    #>> 描画 >>
    
    
if __name__ == '__main__':
    main()
    
    