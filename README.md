## Densly Connected Convolutional Networkds (DenseNet)
#### HUANG, Gao, et al. Densely connected convolutional networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition.

DenseNet is a network architecture where each layer is **directly connected to every other layer** in a feed-forward fashion (within each dense block).
Whereas traditional convolutional networks with L layers have L connections.

![fig1](https://github.com/Oh-Yoojin/DenseNet-keras/blob/master/pictures/fig1.png)

To ensure maximum information flow between layers in the network, connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains **additional inputs from all preceding layers** and passes on its own feature-maps to all subsequent layers.

##### Composite function

Try to understand composite function to Dense block.

![fig3](https://github.com/Oh-Yoojin/DenseNet-keras/blob/master/pictures/fig3.png)

##### Pooling layer

The concatenation operation is not viable when the size of feature-maps changes.

![fig2](https://github.com/Oh-Yoojin/DenseNet-keras/blob/master/pictures/fig2.png)

However, an essential part of convolutional networks is down-sampling layers that change the size of feature-maps. To facilitate down-sampling in DenseNet architecture, divide the network into multiple densely connected dense blocks.

##### Compression
To improve model compactness, reduce the number of feature-maps at transition layers.

* Architecture

![table1](https://github.com/Oh-Yoojin/DenseNet-keras/blob/master/pictures/table1.PNG)

##### Adventage

![table2](https://github.com/Oh-Yoojin/DenseNet-keras/blob/master/pictures/table2.PNG)

###### Accuracy.
DenseNet achieves high accuracy than other traditional architectures.
In this paper, the best results on C10 and C100 are even more encouraging: both are close to 30% lower than FractalNet with drop-path regularization.

###### Parameter Efficiency.
DenseNet utilize parameters more efficiently than alternative architectures.
For example, in 250-layer model only has 15.3M parameters, but it consistently outperforms other models such as FractalNet and Wide ResNets that have more than 30M parameters.

###### Overfitting.
One positive side-effect of the more efficient
use of parameters is a tendency of DenseNets to be less prone to overfitting. We observe that on the datasets without data augmentation, the improvements of DenseNet architectures over prior work are particularly pronounced.
