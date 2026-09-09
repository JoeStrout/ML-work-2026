"""Export the trained AlphaZero net to a raylib-miniscript Matrix blob.

Writes a sequence of MSMX-headered float32 matrices that `assets/lib/aznet.ms`
reads back in order with `readRawData`.

Two transformations happen here so that the MiniScript side stays a plain
sequence of gemms:

1.  **BatchNorm is folded into the preceding convolution.**  Every conv in the
    net has `bias=False`, so a folded layer is exactly `conv(W') + b'` with
    `W' = W * s` and `b' = beta - mean * s`, where `s = gamma / sqrt(var + eps)`.
    Nothing at inference time needs to know BatchNorm existed.

2.  **Convolutions are pre-arranged for im2col.**  Activations live as
    42 cells x C channels (cell = row*COLS + col), so a 3x3 conv is one
    42 x (9*Cin) by (9*Cin) x Cout product.  The weight is laid out with row
    index `k*Cin + ci`, where `k = kh*3 + kw` enumerates the nine (dr, dc)
    offsets in the same order the MiniScript im2col gathers them.

The one genuinely fiddly bit is `p_fc`: PyTorch's `flatten(1)` over a
(2, ROWS, COLS) tensor is channel-major (`ch*42 + cell`), while our activation
layout flattens cell-major (`cell*2 + ch`).  Its weight columns are permuted
here rather than shuffling activations at runtime.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from connect4.az_network import AZNet
from connect4.env import COLS, ROWS, Connect4Env

MAGIC = b"MSMX"
VERSION = 1
DTYPE_FLOAT32 = 2


def write_matrix(buf: bytearray, m: np.ndarray) -> None:
    """Append one MSMX-headered float32 matrix (row-major, little-endian)."""
    if m.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {m.shape}")
    buf += struct.pack("<4sHHii", MAGIC, VERSION, DTYPE_FLOAT32, *m.shape)
    buf += np.ascontiguousarray(m, dtype="<f4").tobytes()


def fold_bn(weight: torch.Tensor, bn: dict, prefix: str, eps: float = 1e-5):
    """Fold a BatchNorm into the conv before it.

    Returns (scaled conv weight, bias vector) as numpy arrays.
    """
    gamma = bn[f"{prefix}.weight"].numpy()
    beta = bn[f"{prefix}.bias"].numpy()
    mean = bn[f"{prefix}.running_mean"].numpy()
    var = bn[f"{prefix}.running_var"].numpy()
    scale = gamma / np.sqrt(var + eps)
    w = weight.numpy() * scale[:, None, None, None]
    b = beta - mean * scale
    return w, b


def conv_to_matrix(w: np.ndarray) -> np.ndarray:
    """(Cout, Cin, 3, 3) or (Cout, Cin, 1, 1) -> (K*Cin, Cout) for im2col gemm.

    Row index is `k*Cin + ci` with `k = kh*3 + kw`, matching the offset order
    the MiniScript im2col uses (dr, then dc, each over -1, 0, 1).
    """
    cout, cin, kh, kw = w.shape
    # (Cout, Cin, kh, kw) -> (kh, kw, Cin, Cout) -> (kh*kw*Cin, Cout)
    return w.transpose(2, 3, 1, 0).reshape(kh * kw * cin, cout)


def build_blob(sd: dict) -> tuple[bytearray, list[str]]:
    buf = bytearray()
    names: list[str] = []

    def emit(name: str, m: np.ndarray) -> None:
        write_matrix(buf, m)
        names.append(f"{name} {m.shape[0]}x{m.shape[1]}")

    # --- stem: conv(2 -> 128, 3x3) + bn + relu ---
    w, b = fold_bn(sd["stem.0.weight"], sd, "stem.1")
    emit("stem.W", conv_to_matrix(w))
    emit("stem.b", b[None, :])

    # --- trunk: 4 residual blocks ---
    n_blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("trunk."))
    for i in range(n_blocks):
        for j in (1, 2):
            w, b = fold_bn(sd[f"trunk.{i}.conv{j}.weight"], sd, f"trunk.{i}.bn{j}")
            emit(f"trunk.{i}.conv{j}.W", conv_to_matrix(w))
            emit(f"trunk.{i}.conv{j}.b", b[None, :])

    # --- policy head: conv(128 -> 2, 1x1) + bn + relu, then fc to COLS ---
    w, b = fold_bn(sd["p_conv.weight"], sd, "p_bn")
    emit("p_conv.W", conv_to_matrix(w))
    emit("p_conv.b", b[None, :])

    # Permute p_fc's input columns from PyTorch's channel-major flatten
    # (ch*42 + cell) to our cell-major one (cell*2 + ch).
    pfc = sd["p_fc.weight"].numpy()               # (COLS, 2*ROWS*COLS)
    cells = ROWS * COLS
    perm = np.array([ch * cells + cell for cell in range(cells) for ch in range(2)])
    emit("p_fc.W", pfc[:, perm].T)                # (84, COLS)
    emit("p_fc.b", sd["p_fc.bias"].numpy()[None, :])

    # --- value head: conv(128 -> 1, 1x1) + bn + relu, fc 42->64->1, tanh ---
    w, b = fold_bn(sd["v_conv.weight"], sd, "v_bn")
    emit("v_conv.W", conv_to_matrix(w))
    emit("v_conv.b", b[None, :])
    emit("v_fc1.W", sd["v_fc1.weight"].numpy().T)
    emit("v_fc1.b", sd["v_fc1.bias"].numpy()[None, :])
    emit("v_fc2.W", sd["v_fc2.weight"].numpy().T)
    emit("v_fc2.b", sd["v_fc2.bias"].numpy()[None, :])

    return buf, names


# --------------------------------------------------------------------------
# A numpy re-implementation of the forward pass, in exactly the form the
# MiniScript will run it.  If this matches PyTorch, the layout and the folding
# are right, and any remaining MiniScript discrepancy is a bug in the port.
# --------------------------------------------------------------------------

def im2col(h: np.ndarray, ksize: int) -> np.ndarray:
    """(cells, C) -> (cells, K*C), zero-padded, offsets in (dr, dc) order."""
    if ksize == 1:
        return h
    c = h.shape[1]
    grid = h.reshape(ROWS, COLS, c)
    out = np.zeros((ROWS, COLS, 9, c), dtype=h.dtype)
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, -dr), min(ROWS, ROWS - dr)
            c0, c1 = max(0, -dc), min(COLS, COLS - dc)
            out[r0:r1, c0:c1, k, :] = grid[r0 + dr:r1 + dr, c0 + dc:c1 + dc, :]
            k += 1
    return out.reshape(ROWS * COLS, 9 * c)


def forward_numpy(mats: list[np.ndarray], obs: np.ndarray) -> tuple[np.ndarray, float]:
    """obs is (2, ROWS, COLS); returns (policy logits, value)."""
    it = iter(mats)
    nxt = lambda: next(it)

    # (2, ROWS, COLS) -> (cells, 2)
    h = obs.reshape(2, ROWS * COLS).T

    w, b = nxt(), nxt()
    h = np.maximum(im2col(h, 3) @ w + b, 0)

    n_blocks = (len(mats) - 2 - 10) // 4
    for _ in range(n_blocks):
        w1, b1, w2, b2 = nxt(), nxt(), nxt(), nxt()
        y = np.maximum(im2col(h, 3) @ w1 + b1, 0)
        y = im2col(y, 3) @ w2 + b2
        h = np.maximum(h + y, 0)

    w, b = nxt(), nxt()
    p = np.maximum(h @ w + b, 0)                  # (cells, 2)
    w, b = nxt(), nxt()
    logits = (p.reshape(1, -1) @ w + b)[0]        # cell-major flatten

    w, b = nxt(), nxt()
    v = np.maximum(h @ w + b, 0)                  # (cells, 1)
    w, b = nxt(), nxt()
    v = np.maximum(v.reshape(1, -1) @ w + b, 0)
    w, b = nxt(), nxt()
    value = float(np.tanh(v @ w + b))
    return logits, value


def read_blob(data: bytes) -> list[np.ndarray]:
    mats, pos = [], 0
    while pos < len(data):
        magic, ver, dt, rows, cols = struct.unpack_from("<4sHHii", data, pos)
        assert magic == MAGIC and ver == VERSION and dt == DTYPE_FLOAT32
        pos += 16
        n = rows * cols
        mats.append(np.frombuffer(data, "<f4", n, pos).reshape(rows, cols).astype(np.float64))
        pos += n * 4
    return mats


def random_positions(n: int, seed: int = 0) -> list[Connect4Env]:
    """Play random legal moves to get a spread of realistic positions."""
    rng = np.random.default_rng(seed)
    envs = []
    while len(envs) < n:
        env = Connect4Env()
        env.reset()
        for _ in range(rng.integers(0, 20)):
            if env.done:
                break
            env.step(int(rng.choice(np.flatnonzero(env.legal_actions()))))
        if not env.done:
            envs.append(env)
    return envs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/az.pt")
    p.add_argument("--out", default="../../raylib-miniscript/assets/az_c4.msmx")
    p.add_argument("--ref", default="../../raylib-miniscript/assets/az_c4_ref.json",
                   help="reference positions + expected outputs, for the MiniScript test")
    p.add_argument("--ref-positions", type=int, default=8)
    args = p.parse_args()

    net = AZNet()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["net"])
    net.eval()
    sd = {k: v.cpu() for k, v in net.state_dict().items()}

    blob, names = build_blob(sd)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"wrote {out}  ({len(blob)/1e6:.2f} MB, {len(names)} matrices)")
    for n in names:
        print(f"    {n}")

    # Verify the folded/reshaped weights against PyTorch itself.
    mats = read_blob(bytes(blob))
    envs = random_positions(args.ref_positions)
    worst_p = worst_v = 0.0
    refs = []
    for env in envs:
        obs = env.observation()
        with torch.no_grad():
            tp, tv = net(torch.from_numpy(obs).unsqueeze(0))
        tp = tp.squeeze(0).numpy().astype(np.float64)
        tv = float(tv.item())
        np_p, np_v = forward_numpy(mats, obs.astype(np.float64))
        worst_p = max(worst_p, float(np.abs(np_p - tp).max()))
        worst_v = max(worst_v, abs(np_v - tv))
        refs.append({
            # board as ROWS*COLS ints, row 0 = bottom, matching env.board
            "board": env.board.astype(int).reshape(-1).tolist(),
            "player": int(env.current_player),
            "logits": [round(x, 6) for x in tp.tolist()],
            "value": round(tv, 6),
        })

    print(f"\nfolded-numpy vs PyTorch: max |dlogit| = {worst_p:.3e}, "
          f"max |dvalue| = {worst_v:.3e}")
    if worst_p > 1e-3 or worst_v > 1e-4:
        raise SystemExit("ERROR: export does not reproduce the PyTorch forward pass")

    ref = Path(args.ref)
    ref.write_text(json.dumps({"rows": ROWS, "cols": COLS, "positions": refs}, indent=1))
    print(f"wrote {ref}  ({len(refs)} reference positions)")


if __name__ == "__main__":
    main()
