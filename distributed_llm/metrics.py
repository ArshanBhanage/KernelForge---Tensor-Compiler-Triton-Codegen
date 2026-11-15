"""Strict parsers: missing/invalid measurements never become fabricated zeroes."""

import math
import statistics


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return float(value)


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(values):
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values) if values else None,
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
    }


def inference_report(raw, requests, expected_world):
    world = raw["world_info"]
    if (
        world["world_size"] != expected_world
        or world["tp_size"] != expected_world
        or world["pp_size"] != 1
    ):
        raise ValueError("reported rank mapping disagrees with requested TP/PP")
    throughput = number(
        raw["performance"]["system_output_throughput_tok_s"], "throughput"
    )
    if throughput == 0:
        raise ValueError("zero throughput is not a successful inference run")
    if not isinstance(requests, list) or not requests:
        raise ValueError("per-request streaming records are required")
    ttft, itl, ids = [], [], set()
    for row in requests:
        if row["id"] in ids:
            raise ValueError("duplicate request ID")
        ids.add(row["id"])
        if row.get("error_tokens", 0):
            raise ValueError("request contains errors")
        start = number(row["start_timestamp"], "start timestamp")
        first = number(row["first_token_timestamp"], "first token timestamp")
        end = number(row["end_timestamp"], "end timestamp")
        count = row["output_tokens"]
        if type(count) is not int or count < 1 or not start <= first <= end:
            raise ValueError("invalid streaming request timing or output count")
        ttft.append((first - start) / 1e6)  # upstream timestamps are nanoseconds
        if count > 1:
            itl.append((end - first) / (count - 1) / 1e6)
    if raw["request_info"]["num_requests"] != len(requests):
        raise ValueError("request report count mismatch")
    return {
        "gpus": expected_world,
        "output_tokens_per_second": throughput,
        "output_tokens_per_second_per_gpu": throughput / expected_world,
        "ttft": summarize(ttft),
        "request_average_itl": summarize(itl),
        "itl_definition": "(last-first token time)/(output_tokens-1), per request; not individual token gaps",
    }


def nccl_rows(text, collective, gpus):
    """Parse standard nccl-tests 13-column out-of-place/in-place FP32 output."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        if len(fields) != 13:
            raise ValueError("unsupported nccl-tests row schema: " + line)
        if fields[2] != "float":
            raise ValueError("expected FP32 nccl-tests results")
        for mode, offset in (("out_of_place", 5), ("in_place", 9)):
            time, algbw, busbw, wrong = fields[offset : offset + 4]
            # Some collectives do not support in-place and report N/A.
            if any(x in ("N/A", "-", "nan") for x in (time, algbw, busbw, wrong)):
                continue
            if int(wrong) != 0:
                raise ValueError("NCCL data validation failed")
            rows.append(
                {
                    "collective": collective,
                    "gpus": gpus,
                    "mode": mode,
                    "size_bytes": int(fields[0]),
                    "count": int(fields[1]),
                    "latency_us": number(float(time), "latency"),
                    "algbw_gbps": number(float(algbw), "algbw"),
                    "busbw_gbps": number(float(busbw), "busbw"),
                }
            )
    if not rows:
        raise ValueError("no valid nccl-tests measurements")
    return rows
