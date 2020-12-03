# Brain_MRI
뇌 MRI 영상을 이용한 종양 인식
# Train, Test data
Kaggle brain MRI 데이터 셋
![train_data](https://user-images.githubusercontent.com/34363323/100235180-63f7ed80-2f6f-11eb-9370-ba30e3c3ccb6.png)


# Model
UNet 기반으로 인풋 아우풋 사이즈가 동일한 크기로 하여 인풋 영상에 대한 segmentation map 출력
![unet_model](https://user-images.githubusercontent.com/34363323/100235425-c224d080-2f6f-11eb-9e02-687e70c1ab99.png)

# Result
![그림1](https://user-images.githubusercontent.com/34363323/100235159-5fcbd000-2f6f-11eb-8231-b37006ddf8c1.png)
![그림2](https://user-images.githubusercontent.com/34363323/100235173-622e2a00-2f6f-11eb-9594-a17af47e18f0.png)
# Reference
<a href="https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation "> Data </a> </br>
<a href="https://www.kaggle.com/monkira/brain-mri-segmentation-using-unet-keras "> Reference code </a> </br>
<a href="https://arxiv.org/abs/1505.04597">Model : U-Net: Convolutional Networks for Biomedical Image Segmentation</a>
