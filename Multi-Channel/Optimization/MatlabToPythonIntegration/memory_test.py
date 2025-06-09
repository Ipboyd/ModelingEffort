"""
huge_batch_epochs.py  -  Minimal network, maximal RAM, many epochs

Set LEAK = True   -> watch memory ratchet upward across epochs
Set LEAK = False  -> memory stays flat after first peak
"""

import sys, torch, psutil, gc

# ---------------- configuration ----------------
device = (
    "cuda" if (len(sys.argv) > 1 and sys.argv[1].lower() == "cuda") else "cpu"
)
EPOCHS      = 5
LEAK        = False     # <- toggle to False to stop the leak
D_IN, D_OUT = 250000, 25000
BATCH       = 128
LR          = 1e-3
# ------------------------------------------------

def rss_mb():
    return psutil.Process().memory_info().rss / 1024 ** 2

def banner(msg):
    print(f"{msg:<30} RSS = {rss_mb():,.0f} MB")

print("=" * 70)
print(f"Running {EPOCHS} epochs on {device.upper()}  (LEAK={LEAK})")
print("=" * 70)

model = torch.nn.Linear(D_IN, D_OUT, bias=True).to(device)
optim = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    banner(f"\nEpoch {epoch}")

    # 1. Create one gigantic batch each epoch
    x      = torch.randn(BATCH, D_IN, device=device)
    target = torch.randn(BATCH, D_OUT, device=device)

    banner("  forward() start")
    y = model(x)

    banner("  loss + backward()")
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    # Intentional leak keeps autograd graph alive
    leak = y if LEAK else y.detach()

    optim.step()
    banner("  after optim.step()")

    optim.zero_grad(set_to_none=True)

    # Clean up
    del x, target, y, loss
    if not LEAK:
        del leak

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    banner("  end-of-epoch cleanup")

print("\nFinished.")
