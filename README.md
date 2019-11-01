# TensorFlowGPU_for_GCE
GCE(Google Compute Engine) 上での TensorFlowGPU 環境構築用プロジェクト

## 環境設定
* Ubuntu 16.04 LTS
* GCE
* CUDA 9.0
* cuNN 7.4
* TensolFlowGPU 1.12.0

> [TensolFlowGPU ビルド設定](https://www.tensorflow.org/install/source#linux)


## 環境構築の手順

### 0. GCPへの登録
 - CCPへアカウント登録
 - VMインススタンスの課金の有効化
 - VMインスタンスの作成
   - vCPU x 24
   - メモリ 90 GB
   - GPU : NVIDIA Tesla T4
   - Boot disk : Ubuntu 16.04 LTS (Size 250GB)
   - ZONE : us-west1-b
   - HTTPトラフィックを許可
 > 参考サイト： [タダでGCPのGPU(NVIDIA Tesla K80)を使ってディープラーニング(NVIDIA DIGITS)する方法](https://qiita.com/SXDSIR2020/items/060806d9fc4366c58d40)

### 1. CUDAとNVIDIAドライバのインストール
VMインスタンスで以下のcommandを実行し，CUDAとドライバをインストールする
```
$ curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
```
```
$ sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
```
```
$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
```
```
$ sudo apt-get update
$ sudo apt-get install cuda-9-0
```

さらに，GPUのパフォーマンスを最適化するために，以下のcommandを実行する
```
$ sudo nvidia-smi -pm 1
```

### 2. cuDNN7.0のインストール
[ここ](https://developer.nvidia.com/developer-program)でdeveloperアカウントを作成し，cuDNNの以下の３つのファイル[ダウンロード](https://developer.nvidia.com/rdp/cudnn-download)する．

#### version : ubuntu 16.04 cuda-9.0 version
 - libcudnn7_7.6.4.38-1+cuda9.0_amd64.deb
 - libcudnn7-dev_7.6.4.38-1+cuda9.0_amd64.deb
 - libcudnn7-doc_7.6.4.38-1+cuda9.0_amd64.deb
<br>

ダウンロードが完了したら，３つのファイルをStarageへアップロードする．この際，バケットnameは``cuda_09``とする．

アップロードが完了したら gsutil command でそのままインスタンスへの転送する．アップロードするディレクトリはお好みでどうぞ．
```
$ cd {UD_LOAD_PATH}
$ gsutil cp gs://cuda_09/libcudnn7_7.6.4.38-1+cuda9.0_amd64.deb .
$ gsutil cp gs://cuda_09/libcudnn7-dev_7.6.4.38-1+cuda9.0_amd64.deb .
$ gsutil cp gs://cuda_09/libcudnn7-doc_7.6.4.38-1+cuda9.0_amd64.deb .
```

転送が完了したらファイルを展開してインストールする．
```
$ sudo dpkg -i libcudnn7_7.6.4.38-1+cuda9.0_amd64.deb
$ sudo dpkg -i libcudnn7-dev_7.6.4.38-1+cuda9.0_amd64.deb
$ sudo dpkg -i libcudnn7-doc_7.6.4.38-1+cuda9.0_amd64.deb
```
### 3. swap file の設定
swap file がない場合，プログラム実行時にメモリリークする可能性がある．
GCEでLinux仮想マシンを作成すると，UbuntuだろうがCentOSだろうが swap file がない状態で仮想マシンが作成される...
<br>
まずは，``free``commandでswapの有無を確認．
```
$ free -m
```
以下のようになっていたらSwap:がゼロになっているので，swapファイルを作成する必要がある．
```
               total        used        free      shared  buff/cache   available
Mem:            581         148          90           0         342         336
Swap:             0           0           0
```
swap file の作成.　swap file の容量はお好みで．（今回は10G）
```
$ sudo fallocate -l 10G /swapfile
$ sudo chmod 600 /swapfile
$ sudo mkswap /swapfile
$ sudo swapon /swapfile
```
swap file の確認
```
$ free -m
```
```
               total        used        free      shared  buff/cache   available
Mem:            581         148          88           0         344         336
Swap:          1023           0        10023
```

**Tips** : 再起動時にスワップファイルを自動マウントするには，``/etc/fstab``に以下を追加する．
```
/swapfile none swap sw 0 0
```

### 4. GPU認識確認とCUDAの設定
CUDAの設定を行う．
```
$ echo "export PATH=/usr/local/cuda-9.0/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
$ source ~/.bashrc
$ sudo /usr/bin/nvidia-persistenced
```

その後、GPUが認識されているかを確認する．
```
$ nvidia-smi
```
以下のようなresponseがあれば，GPUの設定は完了！
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.72       Driver Version: 410.72       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   42C    P0    65W / 149W |      0MiB / 11441MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
### 5. Python環境の構築
Anaconda のダウンロード
```
$ wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
$ sh ./Anaconda3-5.3.1-Linux-x86_64.sh
$ echo ". /home/{USER_NAME}/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
$ source ~/.bashrc
```
Anaondaの環境構築
Pythonのversionと``env_name``はお好みで．
（今回は``tensorflow==1.12.0``を使いたいので，``Python3.6.5``）
```
$ conda create -n kgat python=3.6.5
$ conda activate kgat
```

### 6. tensorflow-gpuのインストール
condaからinstall
```
$ conda install tensorflow==1.12.0
$ conda install tensorflow-gpu==1.12.0
```

以下のプログラムを実行し，``GPU``と出てきたら，tensorflow-gpuがGPUを認識してくれています．
- test.py
```
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```
- OUT
```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 2319180638018740093
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 11324325888
locality {
  bus_id: 1
}
incarnation: 13854674477442207273
physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
]
```

その他，必要なライブラリがあればinstallする．
 - KGAN
```
$ conda install　numpy==1.15.4
$ conda install　scipy==1.1.0 
$ conda install　scikit-learn==0.20.0
```

### 以上！お疲れ様でした〜〜！
