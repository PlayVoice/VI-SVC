# vits-singing-voice-conversion
vits singing voice conversion based on ppg &amp; hubert

VI-SVC model is just VITS without MAS and DurationPredictor. Big data [more and more wave] make things to be interesing!

## Be copied by svc-develop-team/so-vits-svc 
![coarse_f0_1](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/e2f5e5d3-d169-42c1-953f-4e1648b6da37)
## 各位看官：不感兴趣的当个笑话看，感兴趣的用您的智慧去阅读代码，不要被任何一方有偏见的发言带节奏
![coarse_f0_2](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/f3539c83-7c8a-425e-bf20-2c402132f0f4)

![coarse_f0_3](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/f3cee94a-0eeb-4189-b9bb-7043d06e62ef)


```python
import os
import librosa
import pyworld
import utils
import numpy as np
from scipy.io import wavfile


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path):
        x, sr = librosa.load(path, sr=self.fs)
        assert sr == self.fs
        f0, t = pyworld.dio(
            x.astype(np.double),
            fs=sr,
            f0_ceil=800,
            frame_period=1000 * self.hop / sr,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        for index, pitch in enumerate(f0):
            f0[index] = round(pitch, 1)
        return f0

    # for numpy # code from diffsinger
    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(np.int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse
```

使用开源库实现的用于特定场景的独有代码，就不能是原创？在svc-develop-team那里竟1+1=2的常识？

大部分数学推导都由**四则运算**实现，按svc-develop-team的逻辑，所有使用四则运算的数学推导都不构成原创？

这就是svc-develop-team违反开源协议理由，窃取别人代码，还反过来各种贼喊捉贼？

直接拷贝代码文件，都还能说成来自world库的示例，一个人竟然可以无耻到这样的地步，真的是刷新了我的认知。

svc-develop-team说上面代码是world库的示例，请把代码**链接**放出来？

**直接拷贝**代码文件都能说的如此无耻，更何况代码背后还涉及的系统架构、设计思想等没法举证的呢？高知青年就这样？

![coarse_f0_main](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/aa0d4683-318b-47ad-88c1-d6f8950f4ce8)

可以看到，只有在使用从本项目拷贝的代码后，svc-develop-team的项目才**步入正轨**，之前都是瞎搞

但svc-develop-team，在消化融合本项目代码后，将本项目的项目信息删除，理由竟然是相关代码为**world库示例**，那可是直接拷贝的代码

在svc-develop-team的强盗逻辑中，基于开源库实现的功能都不是原创，这就不奇怪他们能将BigVGAN称为他们原创HiFiGAN-with-snake声码器了

## 维权无果后，开发了so-vits-svc-5.0
开发so-vits-svc-5.0某些意义上就是了为维权，证明我能做的比svc-develop-team好

然而svc-develop-team恬不知耻的**再次来拷贝代码**，具体包括so-vits-svc-5.0中的whisper和bigvgan等相关代码

面对svc-develop-team不遵守开源协议和贼喊捉贼的行为，本人在学术圈发文曝光了svc-develop-team的问题

虽然svc-develop-team的Rcell私聊本人达成和解，然而如[NaruseMioShirakana](https://github.com/NaruseMioShirakana)等其他开发者，利用B站用户不懂代码、在B站带节奏造谣

## Rcell对拷贝代码的真实回应

![Rcell](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/8ebb236d-e233-4cea-9359-8e44029b5af5)


# data-sets
KiSing      http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/

PopCS 		  https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md

opencpop 	  https://wenet.org.cn/opencpop/download/

OpenSinger 	https://github.com/Multi-Singer/Multi-Singer.github.io

M4Singer	  https://github.com/M4Singer/M4Singer/blob/master/apply_form.md


CSD 		    https://zenodo.org/record/4785016#.YxqrTbaOMU4

KSS		      https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset

JVS MuSic	  https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music

PJS		      https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus

JUST Song	  https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song


MUSDB18		  https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems

DSD100 		  https://sigsep.github.io/datasets/dsd100.html


Aishell-3 	http://www.aishelltech.com/aishell_3

VCTK 		    https://datashare.ed.ac.uk/handle/10283/2651


# framework

![base_train](/assets/SVC1.png)

![base_infer](/assets/SVC2.png)

![pro_train](/assets/SVC1_pro.png)

![pro_infer](/assets/SVC2_pro.png)

![unix_infer](/assets/SVC2_unix.png)


# train
[VI-SVC](/svc/README.md)

# how to clone your voice
use base model and your voice data to fine tune, just voice data（speech or song） without lables.

# TODO
NSF-VI-SVC based on openai/whisper

https://github.com/PlayVoice/whisper_ppg

