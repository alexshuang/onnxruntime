import numpy as np
import torch
import onnxruntime

model = torch.nn.Sequential(
			torch.nn.LayerNorm(1024)
		).cuda()

x = torch.randn(4, 512, 1024, dtype=torch.float32).cuda()
torch_out = model(x)
print(torch_out.shape, torch_out.dtype, torch_out.device)

torch.onnx.export(model, x, "layer_norm.onnx", export_params=True, opset_version=12,
					input_names=['input'], output_names=['output'],
					dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

ort_session = onnxruntime.InferenceSession("layer_norm.onnx")

ort_input = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
ort_out = ort_session.run(None, ort_input)
print(np.allclose(torch_out.detach().cpu().numpy(), ort_out[0], rtol=1e-03, atol=1e-05))
