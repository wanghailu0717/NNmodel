import onnx 

print([x.op_type for x in onnx.load("resnet18.onnx").graph.node])

