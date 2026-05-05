import argparse
import os

import torch

from nanovllm.layers.attention import store_kvcache


def _build_store_kvcache_debug_case(device: str):
    N, num_heads, head_dim = 3, 2, 4
    num_blocks, block_size = 2, 4
    key = torch.arange(N * num_heads * head_dim, dtype=torch.float32, device=device).reshape(N, num_heads, head_dim)
    value = key + 1000
    k_cache = torch.full((num_blocks, block_size, num_heads, head_dim), -1.0, dtype=key.dtype, device=device)
    v_cache = torch.full((num_blocks, block_size, num_heads, head_dim), -1.0, dtype=value.dtype, device=device)
    slot_mapping = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    return key, value, k_cache, v_cache, slot_mapping


def run_store_kvcache_debug(interpret: bool, debug_kernel: bool, use_pdb: bool):
    if interpret:
        os.environ["TRITON_INTERPRET"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This debug entry expects CUDA tensors. Set up a CUDA device before running it.")

    key, value, k_cache, v_cache, slot_mapping = _build_store_kvcache_debug_case(device)
    slot_indices = slot_mapping.to(dtype=torch.long)

    print(f"device={device}")
    print(f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET', '0')}")
    print(f"key.shape={tuple(key.shape)}, k_cache.shape={tuple(k_cache.shape)}")
    print(f"slot_mapping={slot_mapping.tolist()}")

    if use_pdb:
        breakpoint()

    store_kvcache(key, value, k_cache, v_cache, slot_mapping, debug=debug_kernel)
    torch.cuda.synchronize()

    linear_k_cache = k_cache.view(-1, key.size(1), key.size(2))
    linear_v_cache = v_cache.view(-1, value.size(1), value.size(2))
    written_k = linear_k_cache[slot_indices]
    written_v = linear_v_cache[slot_indices]

    print("written_k:")
    print(written_k.cpu())
    print("written_v:")
    print(written_v.cpu())

    assert torch.equal(written_k, key)
    assert torch.equal(written_v, value)
    print("store_kvcache debug run passed.")


def main():
    parser = argparse.ArgumentParser(description="Debug entry for nanovllm.layers.attention.store_kvcache")
    parser.add_argument("--interpret", action="store_true", help="Enable Triton interpreter mode before kernel launch.")
    parser.add_argument("--debug-kernel", action="store_true", help="Enable device-side Triton debug prints.")
    parser.add_argument("--pdb", action="store_true", help="Break into pdb before launching the Triton kernel.")
    args = parser.parse_args()
    run_store_kvcache_debug(args.interpret, args.debug_kernel, args.pdb)


if __name__ == "__main__":
    main()
