# Multi-Scale Deformable Attention extension

The C++/CUDA sources in this directory are vendored from
[`fundamentalvision/Deformable-DETR`](https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops)
and retain their Apache-2.0 license in `LICENSE`.

The extension is built as `ultralytics.nn.MultiScaleDeformableAttention` when the project is installed with a
PyTorch environment that can find the CUDA toolkit:

```bash
pip install -e . --no-build-isolation
```

Install the intended PyTorch/CUDA build before running this command. To require the build and fail immediately when
PyTorch or CUDA is unavailable, set `YOLOV10_BUILD_MSDA=1`. To install the rest of the project without this optional
extension, set `YOLOV10_BUILD_MSDA=0`.
