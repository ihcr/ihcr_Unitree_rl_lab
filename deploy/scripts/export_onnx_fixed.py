#!/usr/bin/env python3
import argparse, pathlib, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--obs-dim", type=int, default=124)  # 你的 obs 维度
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt)
    out  = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = torch.jit.load(str(ckpt), map_location="cpu")  # 你的 policy.pt 是 TorchScript
    model.eval()

    dummy = torch.zeros(1, args.obs_dim, dtype=torch.float32)  # 固定 [1,124]
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["obs"],           # 官方 OrtRunner 硬编码的输入名
        output_names=["actions"],      # 官方 OrtRunner 硬编码的输出名（注意复数）
        opset_version=args.opset,
        do_constant_folding=True       # 不用 dynamic_axes，保持固定形状
    )
    print("[ok] saved:", out)

if __name__ == "__main__":
    main()