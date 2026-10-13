"""Q17 CPU-only key/value cache accounting. Run: python3 cache_accounting.py"""
def cache_bytes(tokens, layers, kv_heads, head_dim, bytes_per_value=2):
    # K and V, each [tokens, kv_heads, head_dim], for every layer.
    return 2 * tokens * layers * kv_heads * head_dim * bytes_per_value

def check():
    tokens, layers, query_heads, kv_heads, dim = 8192, 24, 32, 8, 128
    mha = cache_bytes(tokens, layers, query_heads, dim)
    gqa = cache_bytes(tokens, layers, kv_heads, dim)
    assert mha == 4 * gqa and gqa > 0
    print(f'PASS: MHA cache={mha/2**20:.1f} MiB; GQA cache={gqa/2**20:.1f} MiB; reduction={mha/gqa:.1f}x')
if __name__ == '__main__': check()
