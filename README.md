# ViTPose + ST-GCN
## Introduction
The main work of this project is replace openpose by ViTPose as the skeleton data input of ST-GCN.  
And here is the **origin implementions**
[Origin ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [Origin ST-GCN](https://github.com/yysijie/st-gcn).  

What's more, here is the **ViTPose without mmlab** [ViTPose_pytorch](https://github.com/jaehyunnn/ViTPose_pytorch).  
The main part of ViTPose in my project were written with reference to this repo.

## Demo result
![xiaorou](resource/info/xiaorou.gif) ![xiaorou](resource/info/xiaorou2.gif) ![ta_chi](resource/info/ta_chi.gif)

## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)
- Other Python libraries can be installed by `pip install -r requirements.txt`
<!-- - FFmpeg (Optional: for demo only), which can be installed by `sudo apt-get install ffmpeg` -->

### Get pretrained models of ST-GCN
Origin ST-GCN repo provided the pretrained model weithts of our **ST-GCN**. The model weights can be downloaded by running the script
```
bash tools/get_models.sh
```
<!-- The downloaded models will be stored under ```./models```. -->
You can also obtain models from [GoogleDrive](https://drive.google.com/drive/folders/1IYKoSrjeI3yYJ9bO0_z_eDo92i7ob_aF) or [BaiduYun](https://pan.baidu.com/s/1jYY5RG4C7hqkKWsUJ0Rzhw?pwd=1266), and manually put them into ```./models```.

### Get pretrained models of ViTPose
> With classic decoder

| Model | Pretrain | Resolution | AP | AR | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViTPose-B | MAE | 256x192 | 75.8 | 81.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py) | [log](logs/vitpose-b.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSMjp1_NrV3VRSmK?e=Q1uZKs) |
| ViTPose-L | MAE | 256x192 | 78.3 | 83.5 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py) | [log](logs/vitpose-l.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSd9k_kuktPtiP4F?e=K7DGYT) |
| ViTPose-H | MAE | 256x192 | 79.1 | 84.1 | [config](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py) | [log](logs/vitpose-h.log.json) | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe) |

## Demo

<!-- To visualize how ST-GCN exploit local correlation and local pattern, we compute the feature vector magnitude of each node in the final spatial temporal graph, and overlay them on the original video. **Openpose** should be ready for extracting human skeletons from videos. The skeleton based action recognition results is also shwon thereon. -->

You can use the following commands to run the demo.

```shell
# with offline pose estimation for image
python vit_image_estimate.py --image_path ${PATH_TO_image} --ckpt_path ${PATH_TO_ViTPose_Pretrained_Model}

# with offline pose estimation for video
python vit_video_estimate.py --video_path ${PATH_TO_video} --ckpt_path ${PATH_TO_ViTPose_Pretrained_Model}

# with offline skeleton recognition
python main.py demo_offline_vit --video ${PATH_TO_VIDEO}
```

## Data Preparation
This project doesn't have ViTPose train module. So we only need data for ST-GCN.
### Data for ST-GCN
We experimented on two skeleton-based action recognition datasts: **Kinetics-skeleton** and **NTU RGB+D**.
Before training and testing, for convenience of fast data loading,
the datasets should be converted to proper file structure. 
You can download the pre-processed data from 
[GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb) or [BaiduYun](https://pan.baidu.com/s/1z3CBsGlBiIkZxvG9YvpySw?pwd=1266)
and extract files with
``` 
cd st-gcn
unzip <path to st-gcn-processed-data.zip>
```
## Testing Pretrained Models

<!-- ### Evaluation
Once datasets ready, we can start the evaluation. -->

To evaluate ST-GCN model pretrained on **Kinetcis-skeleton**, run
```
python main.py recognition -c config/st_gcn/kinetics-skeleton/test.yaml
```
For **cross-view** evaluation in **NTU RGB+D**, run
```
python main.py recognition -c config/st_gcn/ntu-xview/test.yaml
```
For **cross-subject** evaluation in **NTU RGB+D**, run
```
python main.py recognition -c config/st_gcn/ntu-xsub/test.yaml
``` 

<!-- Similary, the configuration file for testing baseline models can be found under the ```./config/baseline```. -->

To speed up evaluation by multi-gpu inference or modify batch size for reducing the memory cost, set ```--test_batch_size``` and ```--device``` like:
```
python main.py recognition -c <config file> --test_batch_size <batch size> --device <gpu0> <gpu1> ...
```
## Training
To train a new ST-GCN model, run

```
python main.py recognition -c config/st_gcn/<dataset>/train.yaml [--work_dir <work folder>]
```
where the ```<dataset>``` must be ```ntu-xsub```, ```ntu-xview``` or ```kinetics-skeleton```, depending on the dataset you want to use.
The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default or ```<work folder>``` if you appoint it.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. For more information, use ```main.py -h```.

Finally, custom model evaluation can be achieved by this command as we mentioned above:
```
python main.py recognition -c config/st_gcn/<dataset>/test.yaml --weights <path to model weights>
```
