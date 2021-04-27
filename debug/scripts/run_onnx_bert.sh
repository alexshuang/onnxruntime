#!/bin/sh

mkdir -p out

CMD="../../build/Release/onnxruntime_training_bert --use_mixed_precision 1 --model_name /workspace/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12 --train_batch_size 128 --mode perf --num_train_steps 1000 --optimizer adam --learning_rate 5e-4 --gradient_accumulation_steps 1 --max_seq_length 128 --max_predictions_per_seq=20"

echo "pmc: L2CacheHit" > input.txt
rocprof -i input.txt --obj-tracking on --timestamp on --stats -o out/res.csv $CMD
