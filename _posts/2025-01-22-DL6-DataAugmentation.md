---
layout: single
title:  "DL-6. Data Augmentation"
---

# Data Augmentation 이란?
- 이미지 데이터를 다양성을 인위적으로 증가시켜 모델의 성능 향상 및 과적합 방지
- 좋은 이미지를 여러 개 얻는 건 비용이 많이 드므로, 원래 이미지를 마구 변형시켜서 학습시킴
- torchvision.transforms와 albumentations. 이 2가지 라이브러리로 진행

## 1. torchvision.transforms와 albumentations
- torchvision.transforms : PyTorch와 같이 사용하기 편리함. 대신 변환 및 증강이 제한적이다.
- albumentations : PyTorch와 잘 호환되진 않지만, 편리한 변환 및 증강이 많이 있음.

- torchvision은 바로 실행시킬 수 있지만, albumentation은 변환 선언 이후 적용의 과정을 거침


```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))

])
dataset = datasets.FashionMNIST(
    root = 'data',
    transform = transform,
    download = True,
    train = True
)

```


```python
plt.imshow(dataset.train_data[1], cmap = 'gray')
```

    c:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\datasets\mnist.py:76: UserWarning: train_data has been renamed data
      warnings.warn("train_data has been renamed data")
    




    <matplotlib.image.AxesImage at 0x1a29ae29130>




![png](/assets/images/2025-01-22-DL6-DataAugmentation_4_2.png)




```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.Normalize(mean =(0.5,), std = (0.5,)),
    ToTensorV2()

])
```

    c:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\albumentations\__init__.py:28: UserWarning: A new version of Albumentations is available: '2.0.4' (you have '2.0.3'). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
      check_for_updates()
    


```python
class AlbumentationDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        image,label = self.dataset[idx]
        if self.transform:
            image = self.transform(image = np.array(image))['image']
        return image, label

train_dataset = datasets.FashionMNIST(root= 'data', download=True, train = True, transform = transform)
```


```python
def show_image(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
```

## 2. 어떤 방식으로 Augmentation 하는가
- 1. 뒤집기(Flip)
- 2. 잘라내기(Crop)
- 3. 이동(Shift), 스케일링(Sclae), 회전(Rotate)
- 4. 색상 조정(ColorJitter)
- 5. 노이즈 추가(GaussNoise)
- 6. GlassBlur
- 7. CLAHE
- 8. CoarseDropout


```python
# 사용할 이미지 들고오기
from PIL import Image
image_path = "example//cat.jpg"
image = Image.open(image_path)
```

### Flip


```python
# 좌우 반전한 이미지
horizontal_flip = transforms.RandomHorizontalFlip(p=1)  # 50% 확률로 좌우 반전
flipped_horizontal_image = horizontal_flip(image)
vertical_flip = transforms.RandomVerticalFlip(p=1)
flipped_vertical_image = vertical_flip(image)
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
show_image(image, 'Original Image')
plt.subplot(1, 3, 2)
show_image(flipped_horizontal_image, 'Horizontally Flipped Image')
plt.subplot(1, 3, 3)
show_image(flipped_vertical_image, 'Vertically Flipped Image')
plt.show()
```


![png](/assets/images/2025-01-22-DL6-DataAugmentation_11_0.png)
    



```python
# 같은 코드지만 albumentation으로 구현하는 경우
image_np = np.array(image)

# 100% 확률로 좌우 및 상하 뒤집기 변환 정의
transform_horizontal_flip_100 = A.HorizontalFlip(p=1.0)
transform_vertical_flip_100 = A.VerticalFlip(p=1.0)

# 데이터 증강 적용 (100%)
flipped_horizontal_100 = transform_horizontal_flip_100(image=image_np)['image']
flipped_vertical_100 = transform_vertical_flip_100(image=image_np)['image']

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
show_image(image_np, 'Original Image')

plt.subplot(1, 3, 2)
show_image(flipped_horizontal_100, '100% Horizontally Flipped Image')

plt.subplot(1, 3, 3)
show_image(flipped_vertical_100, '100% Vertically Flipped Image')
plt.show()
```


![png](/assets/images/2025-01-22-DL6-DataAugmentation_12_0.png)
    


### Crop
- RandomCrop = 이미지의 랜덤 일부를 잘라서 줌. 실행할 때마다 달라짐.
- CenterCrop = 이미지의 정중앙만 잘라서 줌  
둘 다 어느정도를 잘라야 하는지를 height, width로 전달


```python
transform_random_crop = A.RandomCrop(height=80, width=80, p=1.0)  # 랜덤 자르기
transform_center_crop = A.CenterCrop(height=80, width=80, p=1.0)  # 중앙 자르기

random_cropped = transform_random_crop(image=image_np)['image']
center_cropped = transform_center_crop(image=image_np)['image']

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
show_image(image_np, 'Original Image')

plt.subplot(1, 3, 2)
show_image(random_cropped, 'Random Cropped')

plt.subplot(1, 3, 3)
show_image(center_cropped, 'Center Cropped')

plt.show()
```


![png](/assets/images/2025-01-22-DL6-DataAugmentation_14_0.png)
    


### Shift, Scale, Rotate
- ShiftScaleRotate 함수로 한번에 적용 가능
- shift_limit = 이미지 크기의 최대 몇프로까지 이동하는가. 모든 픽셀이 특정 비율만큼 단체 이동
- scale_limit = 이미지 원본 크기의 몇프로까지 변경하는가. 
- rotate_limit = 이미지를 특정 각도만큼 회전


```python
shift_transform = A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0, rotate_limit=0, p =1) #이미지를 원본 크기의 +-30% 범위 내에서 무작위 이동
scale_transform = A.ShiftScaleRotate(shift_limit=0, scale_limit=0.3, rotate_limit=0, p =1) #이미지를 원본 크기의 70% ~ 130%만큼 수정
rotate_transform = A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45, p =1) #이미지를 45도 각도만큼 회전
combined_transform = A.ShiftScaleRotate(shift_limit = 0.3, scale_limit= 0.3, rotate_limit=45 , p =1 ) #모든 작업을 합침

shift_transformed = shift_transform(image=image_np)['image']
scale_transformed = scale_transform(image=image_np)['image']
rotate_transformed= rotate_transform(image=image_np)['image']
combined_transformed = combined_transform(image=image_np)['image']

plt.figure(figsize=(10, 3))

plt.subplot(1, 5, 1)
show_image(image_np, 'Original Image')
plt.subplot(1, 5, 2)
show_image(shift_transformed, 'Shifted')
plt.subplot(1, 5, 3)
show_image(scale_transformed, 'Scaled')
plt.subplot(1, 5, 4)
show_image(rotate_transformed, 'Rotated')
plt.subplot(1, 5, 5)
show_image(combined_transformed, 'Combined')
plt.show()
```

    c:\Users\user\AppData\Local\Programs\Python\Python312\Lib\site-packages\albumentations\core\validation.py:58: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
      original_init(self, **validated_kwargs)
    


![png](/assets/images/2025-01-22-DL6-DataAugmentation_16_1.png)
    


### ColorJitter
- ColorJitter 을 통해 이미지의 밝기, 대비, 채도, 색조 조정 가능
- brightness : 0 ~ 1 사이. 0 : 원본, 1 : 완전히 밝거나 어두움
- contrast : ""
- saturation: 0 ~ 1 사이. 0 : 흑백 이미지, 1 : 색상 강도 최대화
- hue : -0.5 ~ 0.5 사이로 일반적으로 설정 


```python
color_brightness_transform = A.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0, p=1.0)
color_contrast_transform = A.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0, p=1.0)
color_saturation_transform = A.ColorJitter(brightness=0, contrast=0, saturation=0.5, hue=0, p=1.0)
color_hue_transform = A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5, p=1.0)
color_all_transform = A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=1.0)

color_brightness_image = color_brightness_transform(image=image_np)['image']
color_contrast_image = color_contrast_transform(image=image_np)['image']
color_saturation_image = color_saturation_transform(image=image_np)['image']
color_hue_image = color_hue_transform(image=image_np)['image']
color_all_image = color_all_transform(image = image_np)['image']

plt.figure(figsize=(10, 5)) 

plt.subplot(2, 3, 1) 
show_image(image_np, 'Original Image')
plt.subplot(2, 3, 2)  
show_image(color_brightness_image, 'Color brightness Image')
plt.subplot(2, 3, 3) 
show_image(color_contrast_image, 'Color Contrast Image')
plt.subplot(2, 3, 4)  
show_image(color_saturation_image, 'Color Saturation Image')
plt.subplot(2, 3, 5)  
show_image(color_hue_image, 'Color Hue Image')
plt.subplot(2, 3, 6)  
show_image(color_all_image, 'Color all Image')

plt.show()
```


![png](/assets/images/2025-01-22-DL6-DataAugmentation_18_0.png)
    


### GaussianNoise
- 이미지에 노이즈 첨가 가능, 각 픽셀 값에 무작위 값을 추가해서 지지직 거리는 느낌.
- var_limit 을 통해 노이즈의 분산 범위를 정함. Diffusion Model과 같은 맥락


```python
gauss_noise_transform1 = A.GaussNoise(var_limit=(300.00, 400.0), p=1.0) #노이즈의 분산이 300 ~ 400의 값을 지님
gauss_noise_transform2 = A.GaussNoise(var_limit=(1000.00, 1100.0), p=1.0)

gauss_noised_image1 = gauss_noise_transform1(image=image_np)['image']
gauss_noised_image2 = gauss_noise_transform2(image=image_np)['image']

plt.figure(figsize=(10, 5)) 

plt.subplot(1, 3, 1) 
show_image(image_np, 'Original Image')

plt.subplot(1, 3, 2)  
show_image(gauss_noised_image1, 'First Noise Image')

plt.subplot(1, 3, 3)  
show_image(gauss_noised_image2, 'Second Noise Image')

plt.show()
```

    C:\Users\user\AppData\Local\Temp\ipykernel_17472\892153738.py:1: UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
      gauss_noise_transform1 = A.GaussNoise(var_limit=(300.00, 400.0), p=1.0) #노이즈의 분산이 300 ~ 400의 값을 지님
    C:\Users\user\AppData\Local\Temp\ipykernel_17472\892153738.py:2: UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
      gauss_noise_transform2 = A.GaussNoise(var_limit=(1000.00, 1100.0), p=1.0)
    


![png](/assets/images/2025-01-22-DL6-DataAugmentation_20_1.png)
    



```python
import cv2

def show_augmented_images(image_path, transform, n_rows=5, n_cols=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width, _ = image.shape
    crop_size = min(height, width, 450)  # 이미지 크기에 따라 크롭 크기 조정

    # 데이터 증강 파이프라인을 이미지 크기에 맞게 다시 정의
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(1000.00, 1100.0), p=0.5
                    )
    ])

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 20))  # 크기 조정하여 25개 이미지를 표시

    for i in range(n_rows * n_cols):
        augmented = transform(image=image)  # 적용된 데이터 증강
        aug_image = augmented['image']
        
        # 각 subplot에 이미지 표시
        ax = axs[i // n_cols, i % n_cols]
        ax.imshow(aug_image)
        ax.axis('off')
    plt.show()

# 이미지 경로와 증강 파이프라인을 함수에 전달
image_path = "example//cat.jpg"
show_augmented_images(image_path, None)  # 파이프라인 객체를 직접 전달
```

    C:\Users\user\AppData\Local\Temp\ipykernel_17472\1277123024.py:18: UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
      A.GaussNoise(var_limit=(1000.00, 1100.0), p=0.5
    


![png](/assets/images/2025-01-22-DL6-DataAugmentation_21_1.png)
    


### GlassBlur = 이미지의 각 픽셀을 무작위로 이동시킴
- Sigma, max_delta, iteration, always_apply
- Sigma = 흐림의 정도를 제어
- max_delta = 픽셀을 이동시킬 최대 거리
- iteration = 반복 적용



```python
transform = A.GlassBlur(sigma=0.5, max_delta=2, iterations=2, always_apply=True)
    
augmented_image = transform(image=image_np)['image']
    
plt.figure(figsize=(10, 5)) 

plt.subplot(2, 3, 1) 
show_image(image_np, 'Original Image')

plt.subplot(2, 3, 2)  
show_image(augmented_image, 'Glass Blurred Image')

plt.show()
```

    C:\Users\user\AppData\Local\Temp\ipykernel_17472\1349824843.py:1: UserWarning: Argument(s) 'always_apply' are not valid for transform GlassBlur
      transform = A.GlassBlur(sigma=0.5, max_delta=2, iterations=2, always_apply=True)
    


![png](/assets/images/2025-01-22-DL6-DataAugmentation_23_1.png)
    


### CLAHE
- 이미지의 대비를 개선하기 위해 진행
- 이미지를 작은 구역으로 나누고, 그 구역 내에서의 대비를 히스토그램으로 계산한다. 
- clip_limit을 통해 대비 제한을 적용해 극단적인 히스토그램의 빈도수를 제한한다.


```python
transform = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True)
    
augmented_image = transform(image=image_np)['image']
    
plt.figure(figsize=(10, 5)) 

plt.subplot(2, 3, 1) 
show_image(image_np, 'Original Image')

plt.subplot(2, 3, 2)  
show_image(augmented_image, 'Color brightness Image')

plt.show()
```

    C:\Users\user\AppData\Local\Temp\ipykernel_17472\1445347203.py:1: UserWarning: Argument(s) 'always_apply' are not valid for transform CLAHE
      transform = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True)
    


![png](/assets/images/2025-01-22-DL6-DataAugmentation_25_1.png)
    


### CoarseDropout
- 이미지에서 무작위 영역을 선택해 다른 값으로 대체해버림.
- 영역 내의 픽셀은 검정(0) 혹은 평균 값 등으로 대체
- 영역의 개수, 영역의 크기, 대체 값 등을 설정해주면 됨


```python
transform = A.CoarseDropout(max_holes=20, max_height=8, max_width=8, 
                                min_holes=2, min_height=4, min_width=4,
                                fill_value=0, always_apply=True)
    
augmented_image = transform(image=image_np)['image']
    
plt.figure(figsize=(10, 5)) 

plt.subplot(2, 3, 1)
show_image(image_np, 'Original Image')

plt.subplot(2, 3, 2)  
show_image(augmented_image, 'Color CoarseDropout Image')

plt.show()
```

    C:\Users\user\AppData\Local\Temp\ipykernel_17472\932511038.py:1: UserWarning: Argument(s) 'max_holes, max_height, max_width, min_holes, min_height, min_width, fill_value, always_apply' are not valid for transform CoarseDropout
      transform = A.CoarseDropout(max_holes=20, max_height=8, max_width=8,
    


![png](/assets/images/2025-01-22-DL6-DataAugmentation_27_1.png)
    

