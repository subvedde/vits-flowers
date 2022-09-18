# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)

----
## 注意
### 1.torch1.7 以下版本在Windows下进行分布式训练会报错：

AttributeError: module ‘torch.distributed’ has no attribute ‘init_process_group’

#### 报错原因：
torch1.7 以下版本不支持Windows下的分布式训练，在Linux内核才不会报这个错。

#### 解决办法：
方法1：
换成Linux系统运行代码。。。（要是没有条件直接看 方法2）

方法2：
1）、将Windows下的 torch 换成 1.7.0及以上的版本。（1.5~1.8 版本的 torch 代码基本都兼容）

温馨提示：建议离线下载 torch1.12.1+cu113 版本，torchvision0.13.1+cu113版本。 cuda 11.3

###  2.如果用torch 1.7.1版本，windows对pytorch分布式计算支持有问题

RunTimeError：No rendezvous handler for env://

#### 报错原因：
源于多GPU训练，而且windows对pytorch分布式计算支持不够

#### 解决办法：
尝试更新pytorch至1.8。（为了提高 NCCL 稳定性，PyTorch 1.8 将支持稳定的异步错误/超时处理；支持 RPC 分析。此外，还增加了对管道并行的支持，并可以通过 DDP 中的通讯钩子进行梯度压缩。）

###  3.NotImplementedError: Only 2D, 3D, 4D, 5D padding with non-constant padding a

#### 报错原因：
使用来训练的wav文件为双声道,需要改成单声道

#### 解决办法：
使用ffmpeg或者其他工具转成单声道

~~~ 
import  os 
for file in os.listdir("wav"):
   os.system(f"ffmpeg -i {f'wav/{file}'} -ar 22050 -ac 1 {f'newwav/{file}'}") 
   ~~~

###  4.windows平台需要把train.py中的backend='nccl'改成'backend='gloo',因为windows不支持nccl


#### 解决办法：
    dist.init_process_group(backend='gloo', init_method='env://'
                            , world_size=n_gpus, rank=rank)