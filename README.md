# face-mask-detection
Face Mask Detection is a project to determine wether someone have weared mask or not, using deep nerual network.
It contains 3 scripts that provides training new model, testing model with specific images and live face mask detection.  
The gif below shows how live face mask detection works.
<p align="center">
  <img alt="Result gif" align="center" src="https://user-images.githubusercontent.com/33146532/132523023-f9630513-613f-4ab2-a646-b19251635f9b.gif"/>
</p>
This model have trained with 10k face images (5k without mask and 5k with mask - dataset sources are mentioned in dataset section) using 3 layers of 2d CNN as main process(complete summary of model layers is given below).  


```
Model: "sequential"                                               
_________________________________________________________________ 
Layer (type)                 Output Shape              Param #    
================================================================= 
rescaling (Rescaling)        (None, 180, 180, 3)       0          
_________________________________________________________________ 
conv2d (Conv2D)              (None, 180, 180, 16)      448        
_________________________________________________________________ 
max_pooling2d (MaxPooling2D) (None, 90, 90, 16)        0          
_________________________________________________________________ 
conv2d_1 (Conv2D)            (None, 90, 90, 32)        4640       
_________________________________________________________________ 
max_pooling2d_1 (MaxPooling2 (None, 45, 45, 32)        0          
_________________________________________________________________ 
conv2d_2 (Conv2D)            (None, 45, 45, 64)        18496      
_________________________________________________________________ 
max_pooling2d_2 (MaxPooling2 (None, 22, 22, 64)        0          
_________________________________________________________________ 
flatten (Flatten)            (None, 30976)             0          
_________________________________________________________________ 
dense (Dense)                (None, 128)               3965056    
_________________________________________________________________ 
dense_1 (Dense)              (None, 2)                 258        
================================================================= 
Total params: 3,988,898                                           
Trainable params: 3,988,898                                       
Non-trainable params: 0                                           
_________________________________________________________________ 

```
This model have reached fantastic result of 0.9990 accuracy with test dataset( 20% ). moreover, it only takes 10 epochs to reach this result. As the graph below illustrates, from the very first epoch, accuracy is more than 0.98 and by 10th epoch, loss roughly touchs 10e-7 that is a great result.

![loss_acc](https://user-images.githubusercontent.com/33146532/132517900-4be85157-4876-4fb6-8294-c3cd83a8a93e.png)

# dataset
The dataset we have used, is the combination of [Flickr-Faces-HQ (FFHQ) dataset](https://github.com/NVlabs/ffhq-dataset) and [MaskedFace-Net dataset](https://github.com/cabani/MaskedFace-Net). 5k images from each dataset. (sumataion of all datasets is more than 180k images)

