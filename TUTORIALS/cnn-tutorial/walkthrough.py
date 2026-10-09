"""
walkthrough.py -- the worked examples on the slides, computed step by step.

    python walkthrough.py
"""
import math

import numpy as np

from cnn import build_model, shape_table
from conv_numpy import conv2d, conv2d_backward, out_size, pool2d, softmax
from data import load_fashion

np.set_printoptions(precision=3, suppress=True, linewidth=140)


def header(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def as4d(a) -> np.ndarray:
    return np.asarray(a, dtype=float)[None, None]


# ---------------------------------------------------------------- pixels
header("1. AN IMAGE IS A TENSOR; SCALING ONE PIXEL (slide 2)")
d = load_fashion()
print("one image:", d["X_train"][0].shape, "(C, H, W);  a batch of 64:", d["X_train"][:64].shape, "(N, C, H, W)")
print(f"training-split pixel mean {d['mean']:.3f}, std {d['std']:.3f} (after dividing by 255)")
for v in (137, 0, 255):
    print(f"  raw {v:3d} -> {v / 255:.3f} -> ({v / 255:.3f} - {d['mean']:.3f}) / {d['std']:.3f} = {(v / 255 - d['mean']) / d['std']:+.2f}")
print("parameters of ONE fully connected layer with 1,000 hidden units (weights + biases):")
for name, (H, W, C) in (("28x28x1 (Fashion-MNIST)", (28, 28, 1)), ("224x224x3 (ImageNet crop)", (224, 224, 3))):
    print(f"  {name}: inputs {H * W * C:,};  dense layer {(H * W * C + 1) * 1000:,};  64 conv filters 3x3: {64 * (3 * 3 * C + 1):,}")

# ---------------------------------------------------------------- convolution by hand
header("2. CROSS-CORRELATION BY HAND: 5x5 input, 3x3 kernel (slide 3)")
X = np.array([[1, 2, 0, 1, 3],
              [0, 1, 2, 3, 1],
              [2, 0, 1, 1, 0],
              [1, 3, 0, 2, 2],
              [0, 1, 1, 0, 1]])
K = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]])
Y = conv2d(as4d(X), as4d(K))[0, 0]
print("X =\n", X, "\nK =\n", K)
print("Y = X * K (no padding, stride 1): size", out_size(5, 3), "x", out_size(5, 3), "\n", Y.astype(int))
for (i, j) in ((0, 1), (2, 0)):
    patch = X[i:i + 3, j:j + 3]
    terms = " + ".join(f"({k})({p})" for k, p in zip(K.ravel(), patch.ravel()) if k != 0)
    print(f"Y[{i},{j}]: patch\n{patch}\n  = {terms} = {int((patch * K).sum())}")
    print(f"  = (sum of left column) - (sum of right column) = {patch[:, 0].sum()} - {patch[:, 2].sum()}")
flipped = conv2d(as4d(X), as4d(K[::-1, ::-1]))[0, 0]
print("true convolution flips the kernel first; here that gives -Y:", np.array_equal(flipped, -Y))

# ---------------------------------------------------------------- Sobel
header("3. SOBEL KERNEL ON A STEP EDGE (slide 4)")
step = np.array([[0, 0, 1, 1, 1]] * 3)
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
print("image rows: 0 0 1 1 1 (dark on the left, bright on the right)")
print("Sobel-x response along the middle row:", conv2d(as4d(step), as4d(SX))[0, 0].astype(int)[0])
print("Sobel-y response (no change from top to bottom):", conv2d(as4d(step), as4d(SX.T))[0, 0].astype(int)[0])
print("kernel entries sum to", SX.sum(), "-> zero response on any constant region")

# ---------------------------------------------------------------- output size
header("4. OUTPUT SIZE  floor((n + 2p - k) / s) + 1  (slide 5)")
for n, k, s, p, what in ((5, 3, 1, 0, "the example of slide 3"), (5, 3, 1, 1, "'same' padding p = (k-1)/2"),
                         (5, 3, 2, 1, "the drawing on slide 5"), (28, 5, 1, 2, "our conv1"), (14, 5, 1, 0, "our conv2"),
                         (28, 2, 2, 0, "our pool1"), (7, 3, 2, 0, "stride 2, fits exactly"),
                         (8, 3, 2, 0, "stride 2, last column unused"), (224, 7, 2, 3, "ResNet stem"),
                         (227, 11, 4, 0, "AlexNet conv1")):
    print(f"  n={n:3d} k={k:2d} s={s} p={p}: floor(({n} + {2 * p} - {k}) / {s}) + 1 = floor({(n + 2 * p - k) / s:g}) + 1 = "
          f"{out_size(n, k, s, p):3d}   {what}")
x = np.arange(25.0).reshape(1, 1, 5, 5)
print("check with conv2d: 5x5, k=3, s=2, p=1 ->", conv2d(x, np.ones((1, 1, 3, 3)), stride=2, padding=1).shape[2:])

# ---------------------------------------------------------------- receptive field
header("5. RECEPTIVE FIELD  r <- r + (k - 1) j,  j <- j s  (slide 6)")
r, j = 1, 1
print(f"  {'layer':22s} k s   r   jump")
for name, k, s in (("conv 5x5", 5, 1), ("max-pool 2x2", 2, 2), ("conv 5x5", 5, 1), ("max-pool 2x2", 2, 2)):
    r, j = r + (k - 1) * j, j * s
    print(f"  {name:22s} {k} {s}  {r:2d}   {j}")
r = 1
for L in range(1, 4):
    r += 2
    print(f"  {L} stacked 3x3 convs: r = {r}   ({L} x {9} = {9 * L} weights per channel pair; one {r}x{r} kernel: {r * r})")
print("  10 stacked 3x3 convs with stride 1: r =", 1 + 2 * 10, "; with a 2x2 pool after every 2 convs:", end=" ")
r, j = 1, 1
for L in range(10):
    r += 2 * j
    if L % 2 == 1:
        r, j = r + j, j * 2
print("r =", r)

# ---------------------------------------------------------------- channels, parameters
header("6. CHANNELS AND PARAMETERS  C_out (C_in k^2 + 1)  (slides 7, 11)")
for name, ci, co, k in (("conv1", 1, 8, 5), ("conv2", 8, 16, 5), ("RGB example", 3, 64, 3)):
    print(f"  {name:12s} weight ({co}, {ci}, {k}, {k}): {co} * ({ci} * {k * k} + 1) = {co * (ci * k * k + 1):,}")
for name in ("cnn", "mlp", "linear"):
    m = build_model(name)
    print(f"  {name}: {m.num_params():,} parameters")
    if name == "cnn":
        for layer, shape, n in shape_table(m):
            print(f"      {layer:10s} {str(shape):14s} {int(np.prod(shape)):6,d} values {n:8,d} parameters")
print(f"  a dense layer mapping 784 -> 8*28*28 = {8 * 28 * 28} values would need {784 * 6272 + 6272:,} parameters; conv1 has 208")

# ---------------------------------------------------------------- equivariance
header("7. TRANSLATION EQUIVARIANCE IN ONE DIMENSION (slide 8)")
x1 = np.array([0, 1, 3, 1, 0, 0, 0])
k1 = np.array([1, 0, -1])
corr = lambda a: np.array([int((a[i:i + 3] * k1).sum()) for i in range(len(a) - 2)])
x2 = np.roll(x1, 2)
print("x          =", x1, "  ->  x * k          =", corr(x1), "   k =", k1)
print("shift(x,2) =", x2, "  ->  shift(x,2) * k =", corr(x2))
print("the output moved by the same 2 positions:", np.array_equal(corr(x2)[2:], corr(x1)[:-2]))
print("max-pooling the outputs over everything:", corr(x1).max(), "and", corr(x2).max(), "(invariant)")

# ---------------------------------------------------------------- ReLU and pooling
header("8. ReLU AND POOLING (slide 9)")
A = np.array([[1, -2, 0, 3],
              [4, 2, -1, -3],
              [-1, 0, 2, 1],
              [-2, 5, 0, 6]])
R = np.maximum(A, 0)
print("feature map A =\n", A, "\nReLU(A) =\n", R)
print("max-pool 2x2 =\n", pool2d(as4d(R), 2, "max")[0, 0])
print("average-pool 2x2 =\n", pool2d(as4d(R), 2, "mean")[0, 0])
print("global average pool =", R.mean(), "  (sum", R.sum(), "/ 16)")

# ---------------------------------------------------------------- head
header("9. CLASSIFIER HEAD: SOFTMAX AND CROSS-ENTROPY (slide 10)")
logits, label = np.array([2.0, 1.0, 0.0, -1.0]), 1
p = softmax(logits)
print("logits z =", logits, "-> exp =", np.exp(logits), "sum =", round(float(np.exp(logits).sum()), 3))
print("softmax p =", p, "(sums to", round(float(p.sum()), 6), ")")
print(f"true class = {label}: loss = -ln p[{label}] = -ln {p[label]:.3f} = {-math.log(p[label]):.3f}")
g = p.copy()
g[label] -= 1
print("gradient dL/dz = p - onehot =", g)
print(f"uniform guess over 10 classes: loss = ln 10 = {math.log(10):.3f}")

# ---------------------------------------------------------------- backward pass
header("10. BACKWARD PASS OF A CONVOLUTION BY HAND (slide 12)")
Xb = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 1]])
Wb = np.array([[1, -1], [0, 2]])
Yb = conv2d(as4d(Xb), as4d(Wb))
D = np.array([[1, 0], [-1, 2]])                    # dL/dY, handed to us by the layer above
dx, dw, db = conv2d_backward(as4d(D), as4d(Xb), as4d(Wb))
print("X =\n", Xb, "\nW =\n", Wb, "\nY = X * W =\n", Yb[0, 0].astype(int), "\ndelta = dL/dY =\n", D)
print("dL/dW = X * delta (a correlation of the input with the error map) =\n", dw[0, 0].astype(int))
print("  check with conv2d(X, delta):\n", conv2d(as4d(Xb), as4d(D))[0, 0].astype(int))
print("  dL/dW[0,0] = 1*1 + 2*0 + 0*(-1) + 1*2 =", 1 * 1 + 2 * 0 + 0 * -1 + 1 * 2)
print("dL/dX = delta padded by k-1 = 1, correlated with the kernel rotated by 180 degrees =\n", dx[0, 0].astype(int))
print("  check with conv2d(pad(delta), rot180(W)):\n", conv2d(as4d(D), as4d(Wb[::-1, ::-1]), padding=1)[0, 0].astype(int))
print("dL/db = sum of delta =", db[0])

# ---------------------------------------------------------------- batch normalisation
header("11. BATCH NORMALISATION OF ONE CHANNEL (slide 14)")
v = np.array([1.0, 3.0, 5.0, 7.0])
mu, sd = v.mean(), v.std()
print("values of one channel over the batch:", v, " mean", mu, " std", round(float(sd), 3))
print("normalised:", (v - mu) / sd, " then  y = gamma * xhat + beta  with learned gamma, beta per channel")
print("BatchNorm2d(8) has 2 * 8 = 16 learned parameters; dropout has none")
