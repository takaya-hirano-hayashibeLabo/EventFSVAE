# Test Documentation

## test_input_output.py
### 概要
EventFSVAEの入出力関係のみを確認するプログラム.  
同心円上の入力スパイクを入れて, 出力スパイクが出てくるか確認するだけ. 

### 入力次元
|                FSVAE                 |             FSVAE_large              |
| :----------------------------------: | :----------------------------------: |
| [batch_size x channel x 32 x 32 x T] | [batch_size x channel x 64 x 64 x T] |

## test_train.py
### 概要
`test_input_output.py`で入力したスパイクデータを学習して復元できるか確認するプログラム.  
このプログラムで学習できるのは, 入出力が0or1のみ.  
さらに, eventの[0, 1, -1]を学習させたければ, もう少しネットワークを改良する必要がある.  

