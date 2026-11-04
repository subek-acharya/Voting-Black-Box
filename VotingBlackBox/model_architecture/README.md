# model_architecture

Model definitions for the voting / ballot-bubble classifiers that the black-box attacks target.
All classifiers share the same input geometry — grayscale images of size `40×50` with
`num_classes=2` (filled vs. not filled bubble). `ModelFactory` imports these modules and loads the
matching checkpoints.

## Files

- `__init__.py` — package marker (empty).
- `ResNet.py` — `ResNet(nn.Module)` with `BasicBlock`, `LambdaLayer`, and `_weights_init`, plus the
  factory functions `resnet20`, `resnet56`, `resnet164`, `resnet1001`. `ModelFactory` uses
  `resnet20` (input `[1, 1, 40, 50]`, 2 classes).
- `VGG.py` — `VGG(nn.Module)` configurable by name (`"VGG11"`, `"VGG16"`, ...) for `imgH=40,
  imgW=50`, 2 classes, with `_make_layers` / `FigureOutFlattenShape` helpers and a `test()` helper.
- `cait.py` — class-attention image transformer: `CaiT(nn.Module)` built from `Transformer`,
  `Attention`, `FeedForward`, `PreNorm`, and `LayerScale` blocks (ModelFactory config: image
  `40×50`, patch 5, dim 512, depth 16, cls_depth 2, 8 heads).
- `UNet.py` — `UNet(nn.Module)` denoiser/autoencoder used as the preprocessing front-end for the
  `_unet` attack variants.
- `MultiOutputSVM.py` — `BaseSVM(nn.Module)` and `MultiOutputSVM(nn.Module)` (`input_dim =
  1*40*50`); `MultiOutputSVM` loads a base SVM state plus a multi-output head.
- `spiking_resnet_voter.py` — spiking ResNet voter: `SpikingResNetVoter`,
  `SpikingBasicBlockVoter`, and the factory functions `spiking_resnet20_voter` (used by
  `ModelFactory`), `spiking_resnet56_voter`, `_164`, `_1001`. Depends on `spikingjelly`.
- `spiking_vgg_voter.py` — spiking VGG voter: `SpikingVGGVoter` and the factory functions
  `spiking_vgg11_voter`, `spiking_vgg11_bn_voter`, `spiking_vgg13(_bn)_voter`,
  `spiking_vgg16_voter`, `spiking_vgg16_bn_voter` (used by `ModelFactory`), `_19(_bn)`. Depends on
  `spikingjelly`.

## How it is used

`ModelFactory` builds each architecture from these modules and wraps the spiking voters in
`SNNWrapper` (time dimension handling) and, for the `_unet` experiments, wraps a classifier with
the UNet denoiser via `UNetModelWrapper`. The spiking voters require the `spikingjelly` package.
