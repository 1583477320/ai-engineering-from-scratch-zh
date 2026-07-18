"""批处理 vs 同步成本对比。"""


def compare_batch_sync(documents, input_tokens, output_tokens,
                       input_rate=3.0e-6, output_rate=15.0e-6, batch_discount=0.5):
    sync = documents * (input_tokens * input_rate + output_tokens * output_rate) / 1e6 * documents
    sync = (documents * input_tokens * input_rate + documents * output_tokens * output_rate) / 1e6

    first_write = input_tokens * input_rate / 1e6
    cached = first_write + (documents - 1) * input_tokens * input_rate * 0.1 / 1e6
    cached += documents * output_tokens * output_rate / 1e6

    batch_cached = cached * (1 - batch_discount)

    return {"sync": sync, "cached": cached, "batch_cached": batch_cached,
            "savings": 1 - batch_cached / sync}


if __name__ == "__main__":
    r = compare_batch_sync(50000, 4000, 200)
    print(f"同步无缓存: ${r['sync']:.0f}")
    print(f"同步有缓存: ${r['cached']:.0f}")
    print(f"批处理+缓存: ${r['batch_cached']:.0f}")
    print(f"节省: {r['savings']:.1%}")
