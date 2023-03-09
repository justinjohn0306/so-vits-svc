# SoftVC VITS Singing Voice Conversion

## SoVITS has stopped updating and officially Archived
Some other related warehouses for follow-up maintenance and other work:
+ [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc): The svc community is ready to take over and maintain sovits. If you have pr and issue, you can submit them here
+ [SoftVitsResearch](https://github.com/NaruseMioShirakana/SoftVitsResearch): Used to make some fancy functions (mainly for Onnx-MoesS)

## Terms of use
1. Please solve the authorization problem of the data set by yourself. Any problems caused by using unauthorized data sets for training, you must bear full responsibility and all consequences, and have nothing to do with sovits!
2. Any sovits-based video published to the video platform must clearly indicate in the introduction the input source singing and audio used for voice changer conversion, for example: use the video/audio released by others, and use the separated human voice as input If the source is converted, a clear link to the original video and music must be given; if the input source is converted using your own vocals or voice synthesized by other singing voice synthesis engines, you must also explain it in the introduction.
3. Any copyright infringement caused by the input source shall bear full responsibility and all consequences. When using other commercial singing voice synthesis software as the input source, please make sure to abide by the terms of use of the software. Note that the terms of use of many singing voice synthesis engines clearly indicate that they cannot be used for input source conversion!


## Model Introduction
Singing voice conversion model, use [Content Vec](https://github.com/auspicious3000/contentvec) to extract content features, input visinger2 model to synthesize target voice

### 4.0 v2 version update content
+ The model architecture is completely modified to [visinger2](https://github.com/zhangyongmao/VISinger2) architecture
+ Others are exactly the same as 4.0
### 4.0 v2 version features
+ There is a certain improvement compared to 4.0 in some scenes (for example, breathing sound and current sound in some scenes)
+ But there are also some scenarios where the effect is somewhat backward. For example, the effect trained on the Maolei data is not as good as 4.0, and in some cases it will synthesize a very ghostly sound
+ As for the old one or v2, you can try the following demo and the demo on the 4.0 branch to compare and decide
+ 4.0-v2 is the last version of sovits, there will be no more updates after that, sovits will be archived after the basic verification has no major bugs

Online demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/innnky/sovits4.0-v2)

## Notice
+ The entire process of 4.0-v2 is the same as 4.0, and the environment is the same as 4.0. The preprocessed data and environment of 4.0 can be used directly
+ The difference from 4.0 is:
   + The model **completely** is not universal, the old model cannot be used, and the base model needs to use a new base model, please make sure you load the correct base model or the training time will be extremely long!
   + The structure of the config file is very different. Do not use the old config. If you are using the 4.0 dataset, you only need to execute the preprocess_flist_config.py step to generate a new config

## Pre-downloaded model files
+ contentvec: [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
   + placed in the `hubert` directory
+ Pre-trained base model file: [G_0.pth](https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/G_0.pth) and [D_0.pth](https:// huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/D_0.pth)
   + placed in the `logs/44k` directory
   + The pre-trained base model training data set includes Yunhao, Jishuang, Huiyu·Xing AI Paimeng, Lingdi Ningning, covering the common vocal range of male and female girls, which can be considered as a relatively common base model
```shell
# One-click download
# contentvec
# Since the network disk provided by the author does not have a direct link, it needs to be manually downloaded and placed in the hubert directory
# G and D pre-training model:
wget -P logs/44k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/G_0.pth
wget -P logs/44k/ https://huggingface.co/innnky/sovits_pretrained/resolve/main/sovits4.0-v2/D_0.pth

```

**Training Notebook**: <a href="https://colab.research.google.com/github/justinjohn0306/so-vits-svc/blob/4.0-v2/Sovits4_0_v2_One_click_training_inference_script.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> and follow the instructions.

The readme in the latter part is the same as 4.0, no change

## Dataset preparation
Just put the dataset into the dataset_raw directory with the following file structure
```shell
dataset_raw
├───speaker0
│ ├───xxx1-xxx1.wav
│ ├────...
│ └───Lxx-0xx8.wav
└───speaker1
     ├───xx2-0xxx2.wav
     ├────...
     └───xxx7-xxx007.wav
```


## Data preprocessing
1. Resample to 44100hz

```shell
python resample.py
  ```
2. Automatically divide training set, verification set, test set and automatically generate configuration files
```shell
python preprocess_flist_config.py
```
3. Generate hubert and f0
```shell
python preprocess_hubert_f0.py
```
After performing the above steps, the dataset directory is the preprocessed data, you can delete the dataset_raw folder


## train
```shell
python train.py -c configs/config.json -m 44k
```
Note: The old model will be automatically cleared during training, and only the latest 3 models will be kept. If you want to prevent overfitting, you need to manually back up the model record points, or modify the configuration file keep_ckpts 0 to never clear

## reasoning
use [inference_main.py](inference_main.py)

As of now, the 4.0 usage method (training, reasoning) is exactly the same as 3.0, without any changes (command line support is added for reasoning)

```shell
# example
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物语-src.wav" -t 0 -s "nen"
```
required field
+ -m, --model_path: model path.
+ -c, --config_path: configuration file path.
+ -n, --clean_names: list of wav file names, placed under the raw folder.
+ -t, --trans: pitch adjustment, support positive and negative (semitones).
+ -s, --spk_list: synthesize target speaker names.

Optional section: see next section
+ -a, --auto_predict_f0: Voice conversion automatically predicts the pitch, do not enable this when converting singing voices, it will seriously go out of tune.
+ -cm, --cluster_model_path: The path of the cluster model, if there is no training cluster, just fill it in.
+ -cr, --cluster_infer_ratio: Clustering scheme ratio, range 0-1, if no clustering model is trained, fill in 0.

## Optional
If the previous effect is satisfactory, or if you don’t understand what is going on below, then the following content can be ignored without affecting the use of the model. (These options have relatively little impact and may have some effect on some specific data, but most of the cases seem to be less obvious),
### Automatic f0 prediction
The 4.0 model training process will train an f0 predictor. For voice conversion, you can enable automatic pitch prediction. If the effect is not good, you can also use manual, but please do not enable this function when converting singing voices! ! ! It will be seriously out of tune! !
+ Set auto_predict_f0 to true in inference_main
### Cluster Timbre Leakage Control
Introduction: The clustering scheme can reduce the timbre leakage, making the model trained more like the target timbre (but it is not particularly obvious), but the simple clustering scheme will reduce the model's articulation (it will be inarticulate) (this is obvious) , this model adopts the fusion method,
You can linearly control the ratio of the clustering scheme and the non-clustering scheme, that is, you can manually adjust the ratio between "like the target tone" and "clear articulation" to find a suitable compromise point.

Using the existing steps before clustering does not require any changes, only an additional clustering model needs to be trained, although the effect is relatively limited, but the training cost is relatively low
+ Training process:
   + Use a machine with better CPU performance for training. According to my experience, it takes about 4 minutes to train each speaker on Tencent Cloud 6-core CPU to complete the training
   + Execute python cluster/train_cluster.py, the output of the model will be in logs/44k/kmeans_10000.pt
+ Reasoning process:
   + Specify cluster_model_path in inference_main
   + Specify cluster_infer_ratio in inference_main, 0 means no clustering at all, 1 means only clustering, usually set to 0.5

## Onnx export
Use [onnx_export.py](onnx_export.py)
+ Create a new folder: `checkpoints` and open it
+ Create a new folder in the `checkpoints` folder as the project folder, and the folder name is your project name, such as `aziplayer`
+ Rename your model to `model.pth`, rename the configuration file to `config.json`, and place it in the `aziplayer` folder you just created
+ Change `"NyaruTaffy"` in `path = "NyaruTaffy"` in [onnx_export.py](onnx_export.py) to your project name, `path = "aziplayer"`
+ run [onnx_export.py](onnx_export.py)
+ After the execution is completed, a `model.onnx` will be generated in your project folder, which is the exported model
    ### Onnx model supported UI
    + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)
+ I removed all the training functions and all complicated transpositions, not a single line, because I think you only know that you are using Onnx if you remove these things
+ Note: Please use the model provided by MoesS for the Hubert Onnx model, which cannot be exported at present (Hubert in fairseq has many operators that onnx does not support and things that involve constants, and an error will be reported when exporting or the exported model input and output shape and The results are problematic)
[Hubert4.0](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel)
