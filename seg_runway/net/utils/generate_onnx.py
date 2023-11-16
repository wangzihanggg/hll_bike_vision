import torch
from torch import nn
from detect_vehicle_line.net.unet_big import UNetBig
if __name__ == '__main__':
    model = UNetBig(in_channels=3, out_channels=3, WithActivateLast=False)
    model.eval()
    input = torch.randn(4, 3, 256, 256)
    input_name = ['input']
    output_name = ['output']
    torch.onnx.export(model, input,
                      "../onnx/UnetBigResult.onnx",
                      input_names=input_name,
                      output_names=output_name,
                      verbose=True,
                      opset_version=11
                      )
    print("success")
