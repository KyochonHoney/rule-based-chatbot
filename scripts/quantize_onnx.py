#!/usr/bin/env python
import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="in_path", required=True, help="input ONNX model path (.onnx)")
parser.add_argument("--out", dest="out_path", required=True, help="output INT8 ONNX path (.onnx)")
parser.add_argument("--per_channel", action="store_true", help="enable per-channel quant")
args = parser.parse_args()

inp = Path(args.in_path)
out = Path(args.out_path)
out.parent.mkdir(parents=True, exist_ok=True)

print(f"[quantize] {inp} → {out} (INT8 dynamic, per_channel={args.per_channel})")
quantize_dynamic(
    model_input=str(inp),
    model_output=str(out),
    weight_type=QuantType.QInt8,
    per_channel=args.per_channel,
    reduce_range=False,
)
print("[ok] quantized")
