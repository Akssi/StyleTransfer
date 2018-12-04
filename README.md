# Style Transfer
This the implementation of style transfer based on the ReCoNet paper [arXiv](https://arxiv.org/abs/1807.01197)

Training scripts were done by me according to multiple papers (see references)

The script train.py trains the network using a single style image for style refrence.  
The script train_multipleStyle.py trains the network using a dataset for both the content images and the style images.  
The script train_temporalLoss.py trains the network with the additional temporal loss function.  

The savedModel folder contains a few saved weights for diffrent style and for diffrent parameter values.

# Datasets
The training for this network was done using a two diffrent dataset.  
[COCO Dataset](http://cocodataset.org/#home)  
[Monkaa Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

# Results
Here are the stylized versions of the following picture according to multiple style references.  
<img align="center" src="https://drscdn.500px.org/photo/215045239/q%3D80_m%3D1500/v2?user_id=13128095&webp=true&sig=3d293889b1d59822df2d8fb38072bffdf5feddac4ecc426845aa4d33f5b7fd38" alt="Source Image" width="25%">  
<br/><br/>
<img align="left" src="models/style/udnie.jpg" alt="Udnie - Reference" width="20%">
<img align="left" src="models/style/color.jpg" alt="Rain - Reference" width="20%">
<img align="left" src="models/style/mosaic.jpg" alt="Mosaic - Reference" width="20%">
<img align="left" src="models/style/composition.jpg" alt="Composition - Reference" width="20%">  
<br/><br/><br/><br/><br/>
<br/><br/><br/>
<img align="left" src="Results/Udnie_stylized.png" alt="Udnie - Reference" width="20%">
<img align="left" src="Results/Colors_stylized.png" alt="Colors - Reference" width="20%">
<img align="left" src="Results/Mosaic_stylized.png" alt="Mosaic - Reference" width="20%">
<img align="left" src="Results/Composition_stylized.png" alt="Composition - Reference" width="20%">



