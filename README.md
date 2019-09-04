## Citation
```
@inproceedings{huang2017adain,
  title={Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization},
  author={Huang, Xun and Belongie, Serge},
  booktitle={ICCV},
  year={2017}
}
```
[[Github]](https://github.com/xunhuang1995/AdaIN-style)

## Download

You need to download models from this link and put them in the model directory.
[Model](https://drive.google.com/drive/folders/1LTeHsU3Wj4gqMvY7HxEkmLfLK7gmOwfe?usp=sharing)

You can download the content and style images from this link.
[data](https://drive.google.com/drive/folders/1ajwC9PJkuILf81rU97GdllWr2bmKrUz6?usp=sharing)

## Usage
### Basic usage
Use `--contentImage` and `--styleImage` to provide the respective path to the content and style image, for example:
```
pyhton solution.py --contentImage input/content/chicago.jpg --styleImage input/style/ashville.jpg
```
  
### Content-style trade-off
Use `--alpha` to adjust the degree of stylization. It should be a value between 0 and 1 (default).

### Style interpolation
It is possible to interpolate between several styles using `--styleInterpWeights ` that controls the relative weight of each style. Note that you also need to set the `--interpolate ` to True and provide the same number of style images separated by commas. Example usage:
```
python solution.py 
--interpolate True \
--contentImage input/content/avril.jpg \
--styleImage input/style/encampo.jpg,input/style/ashville.jpg \
--styleInterpWeights 0.2,0.8
```

