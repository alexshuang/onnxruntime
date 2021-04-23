import numpy as np
import torch
import onnxruntime
import timeit

x = torch.randn(4, 512, 1024, dtype=torch.float32)
ort_session = onnxruntime.InferenceSession("layer_norm.onnx")
ort_input = {ort_session.get_inputs()[0].name: x.cpu().numpy()}

ort_out = ort_session.run(None, ort_input)

num_loops = 5
num_iters = 1000
ort_out = ort_session.run(None, ort_input)
elapseds = [timeit.timeit(lambda: ort_session.run(None, ort_input), number=num_iters) for _ in range(num_loops)]
elapsed = np.median(elapseds)

print(ort_out[0].shape)
print(f"elapsed: {elapseds}")
print(f"{elapsed/num_iters*1000:.5f} ms per iter, {num_iters} iters, {num_loops} loops")
