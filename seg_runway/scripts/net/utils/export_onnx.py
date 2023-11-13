import torch
import torchvision
import onnxsim
import onnx
import onnxruntime as ort
import numpy as np
import os

cur_dir = os.path.split(os.path.realpath(__file__))[0]
print(cur_dir)
model_save_path = os.path.realpath(
    os.path.join(os.path.dirname(cur_dir), "model"))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

net_name = "googlenet"

onnx_name = "{0}/{1}.onnx".format(model_save_path, net_name)
onnx_simplify_name = "{0}/{1}_simplify.onnx".format(model_save_path, net_name)

dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
model = torchvision.models.googlenet(pretrained=True).cuda()
# model.save()

input_names = ["data"] # 注意这个名字也要和 trt 加载时候的名字一样
output_names = ["prob"]

# 改函数可以有更多的参数配置，比如是否设置为动态输入等，是否压缩常量
torch.onnx.export(model, 
                  dummy_input, # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                  onnx_name, 
                  verbose=True, # 是否以字符串的形式显示计算图
                  input_names=input_names,
                  output_names=output_names)

# simplify onnx 简化 onnx，例如讲一些 CBR 结构合成一个
print("##############simplify onnx#####################")
onnx_model = onnx.load(onnx_name)  # load onnx model​
model_simp, check = onnxsim.simplify(onnx_model) 
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_simplify_name)

# check onnx 检验 onnx 模型是否合法
print("##############check onnx#####################")
onnx_simplify = onnx.load(onnx_simplify_name)  # load onnx model​
onnx.checker.check_model(onnx_simplify)
# print(onnx.helper.printable_graph(onnx_simplify.graph))

print("##############inference onnx#####################")
ort_session = ort.InferenceSession(onnx_simplify_name)
outputs = ort_session.run(
    None, # output_names 输出名字
    {"data": np.ones(shape=[1, 3, 224, 224]).astype(np.float32)},
)
print(outputs[0])
