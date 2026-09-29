import re, subprocess, sys, matplotlib.pyplot as plt

def set_n(path, n):
    text = open(path).read()
    open(path,"w").write(re.sub(r"let n = \d+usize;", f"let n = {n}usize;", text))

sizes = [256, 512, 1024, 1536, 2048]
results = {}
for n in sizes:
    # edit n in source temporarily using sed
    set_n("/Users/stevens/gemm-rust/src/main.rs", n)
    out = subprocess.run(["cargo", "run", "--release", "--manifest-path", "/Users/stevens/gemm-rust/Cargo.toml"],
                         capture_output=True, text=True, cwd="/Users/stevens/gemm-rust")
    # Extract all lines; find Rayon row-block explicitly
    rayon_line = None
    for ln in out.stdout.splitlines():
        if "Rayon" in ln:
            rayon_line = ln
    if rayon_line:
        m = re.search(r"Rayon.*?:\s+(.+?) ", rayon_line)
    if m:
        dur_str = m.group(1)
        # parse duration to seconds (e.g., 70ms)
        val_str = m.group(1)
        val = float(re.search(r"([0-9]+\.?[0-9]*)", val_str).group(1))
        unit = re.search(r"(ms|µs|s)", dur_str)
        sec = val/1e3 if unit and "ms" in unit.group(1) else val/1e6 if unit and "µs" in unit.group(1) else val
        gflops = (2*n**3) / sec / 1e9
        results[n] = gflops
        print(f"n={n}: {gflops:.1f} GFLOPS ({dur_str})")
    else:
        print(f"n={n}: parse error: {line}")

# restore source
set_n("/Users/stevens/gemm-rust/src/main.rs", 1024)

# plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(list(results.keys()), list(results.values()), marker='o', label='Rayon 2D tile')
ax.axhline(288, color='r', linestyle='--', label='Peak i9-10910 AVX2 (288 GFLOPS)')
ax.set_xlabel('Matrix size n')
ax.set_ylabel('Performance (GFLOPS)')
ax.set_title('GEMM Roofline (Rayon row-block / 2D tile)')
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig('/Users/stevens/gemm-rust/benchmark/roofline.pdf')
print("Saved /Users/stevens/gemm-rust/benchmark/roofline.pdf")
