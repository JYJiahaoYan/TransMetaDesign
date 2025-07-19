# Data for Model


## Data Description

- **FDTD_file**: Used for optical response simulation in Lumerical FDTD, generating the required dataset
- **data_gen**: Used to generate parameter sets, generating datasets that meet conditions according to the number and range of parameters, for collecting spectral data in FDTD simulations

### Data Overview

To cover the majority of common structural shapes, the following seven shapes have been designed:
- **Circle**: One of the most common shapes, which collects optical responses corresponding to different parameters by designing the major axis, minor axis, rotation angle, and periodicity.
- **Rectangle**: Another common shape that collects optical responses corresponding to different parameters by designing length, width, rotation angle, and periodicity.
- **Double Rectangle**: Controls the optical response using two rectangles, allowing for the collection of richer optical response information, such as the BIC peak in the infrared range.
- **Double Ellipse**: Similar to double rectangles, it uses two ellipses to control the optical response and can gather richer optical response information, such as the BIC peak in the infrared range.
- **Ring**: One of the most common shapes, which controls the general ring shape through inner diameter and outer diameter, and adjusts the ring by varying the starting and ending angles to achieve more diverse optical responses.
- **Cross**: A common shape that offers a rich degree of freedom to generate diverse optical responses.
- **Lack Rectangle**: A more refined control of rectangles, adding more degrees of freedom on top of traditional rectangles to produce more diverse optical responses.

<p float="left">
  <img src="img\circle.png" alt="图片1" width="200" /><img src="img\rec.png" alt="图片2" width="200" /><img src="img\double_rec.png" alt="图片3" width="200" /><img src="img\double_ellipse.png" alt="图片4" width="200" />
</p>
<p float="left">
  <img src="img\ring.png" alt="图片5" width="200" />
  <img src="img\cross.png" alt="图片6" width="200" />
  <img src="img\lack_rec.png" alt="图片7" width="200" />
</p>

**Parameters**：

| Shape | Params |
|---------|---------|
| circle  | a,b,φ,Px,Py  |
| rec  | W,L,φ,Px,Py  |
| double_rec  | W1,L1,W2,L2,φ,Px,Py|
| double_ellipse  | a,b,θ,φ,Px,Py  |
| ring  | R,r,θ,φ,Px,Py  |
| cross  | W1,L1,W2,L2,offset,φ,Px,Py  |
| lack_rec  | W,L,α,β,γ,φ,Px,Py  |


