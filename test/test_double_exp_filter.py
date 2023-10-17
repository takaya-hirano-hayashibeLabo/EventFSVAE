"""
【プログラム作成理由】
FSVAEの出力を0,1,-1にしたら上手く学習できなかった
原因は損失関数の二重指数関数にあるのでは？と思って,
二重指数関数に-1を入れたときの挙動を調べてみる.

【調査結果】
２重指数関数の`dt,td,tr`の値によって大きくグラフの形が変わる.  
それぞれの値が大きすぎると, 出力がほぼ0になってしまう.  
教科書に載ってたデフォルトの値が良さげな感じがする.  

デフォルト値
|  dt   |  td   |  tr   |
| :---: | :---: | :---: |
| 1e-4  | 1e-2  | 5e-3  |
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DoubleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        self.N = N
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = np.zeros(N)
        self.hr = np.zeros(N)
        
    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt
        hr = self.hr*(1-self.dt/self.td) + spike/(self.tr*self.td)
        self.r = r
        self.hr = hr
        return r
    
    
def main():
    
    T=16
    
    spike=np.where(
        np.random.uniform(0,1,size=(T,1))>0.6,
        1,0
    )
    negative_idx=[int(T/(i+2)) for i in range(5)]
    spike[negative_idx]=-1
    
    double_syn=DoubleExponentialSynapse(N=1)
    double_syn.initialize_states()
    
    r=np.zeros_like(spike)
    for i in range(T):
        r[i]=double_syn(spike[i])
    
    fig,ax=plt.subplots(1,2)
    ax[0].plot(spike)
    ax[1].plot(r)
    
    fig.savefig(f"{str(Path(__file__).parent)}/test_double_exp_filter/in_out.png")
    
if __name__=="__main__":
    main()