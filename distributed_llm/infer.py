"""Real TensorRT-LLM generation. Invoke once; TensorRT-LLM owns worker ranks."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--output", default="generation.json")
    args = p.parse_args()
    if args.max_tokens <= 0:
        p.error("max-tokens must be positive")
    import torch

    if torch.cuda.device_count() < args.tp:
        p.error("not enough visible CUDA devices for requested tensor parallelism")
    from tensorrt_llm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, pipeline_parallel_size=1)
    try:
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        results = [
            {
                "prompt": output.prompt,
                "text": output.outputs[0].text,
                "token_ids": list(output.outputs[0].token_ids),
            }
            for output in llm.generate(args.prompt, params)
        ]
        Path(args.output).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "tensor_parallel_size": args.tp,
                    "outputs": results,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
