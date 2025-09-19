
import argparse, csv, json, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def stage_banner(msg: str):
    print("\n" + "="*80)
    print(msg)
    print("="*80 + "\n")

def run(cmd):
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)

def read_grid_csv(variant: str):
    path = ART / f"tuning_{variant}_act_lr_dropout.csv"
    if not path.exists():
        raise FileNotFoundError(f"Grid CSV not found: {path}")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            # coerce numerics if present
            for k in ("best_val_acc","best_val_loss","duration_sec","lr","dropout"):
                if k in r and r[k] not in (None, ""):
                    try:
                        r[k] = float(r[k])
                    except Exception:
                        pass
            rows.append(r)
    return rows

def run_pipeline(variant: str, epochs: int, patience: int, topk: int, flip_p: float, height: int, width: int):
    # Fixed settings (no size/grayscale search)
    H, W = height, width
    stage_banner(f"[{variant.upper()}] GRID — ACTIVATION × LR × DROPOUT ({H}×{W}, RGB)")

    # ---- Stage: ACT × LR × DROPOUT ----
    cmd = [sys.executable, "tuning.py",
           "--variant", variant, "--stage", "act_lr_dropout",
           "--epochs", str(epochs), "--patience", str(patience),
           "--height", str(H), "--width", str(W)]
    run(cmd)

    # Load grid results & pick Top-K
    rows = read_grid_csv(variant)
    rows = sorted(rows, key=lambda r: float(r.get("best_val_acc", 0.0)), reverse=True)
    finalists = rows[:topk]

    # Write a concise topK summary (what we're about to run)
    topk_csv = ART / f"top{topk}_summary_{variant}.csv"
    with open(topk_csv, "w", newline="") as f:
        fieldnames = ["rank","variant","H","W","activation","lr","dropout","flip_p","grid_best_val_acc","grid_best_val_loss"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(finalists, start=1):
            w.writerow({
                "rank": i, "variant": variant, "H": H, "W": W,
                "activation": r.get("activation"), "lr": r.get("lr"),
                "dropout": r.get("dropout"), "flip_p": flip_p,
                "grid_best_val_acc": r.get("best_val_acc"), "grid_best_val_loss": r.get("best_val_loss")
            })
    print(f"WROTE Top-{topk} summary -> {topk_csv}")

    # ---- FINAL: run each finalist with main.py ----
    stage_banner(f"[{variant.upper()}] FINAL — run Top-{topk}")
    start_all = time.time()
    final_rows = []
    for i, r in enumerate(finalists, start=1):
        act = str(r.get("activation"))
        lr = str(r.get("lr"))
        do = str(r.get("dropout"))
        tag = f"top{i}_{act}_lr{lr}_do{do}".replace(" ", "")
        cmd = [sys.executable, "main.py",
               "--variant", variant,
               "--activation", act, "--lr", lr, "--dropout", do,
               "--epochs", str(max(epochs, 40)), "--patience", str(max(patience, 8)),
               "--height", str(H), "--width", str(W), "--flip_p", str(flip_p),
               "--tag", tag]
        run(cmd)
        # Record a minimal line; detailed metrics should be in main's outputs (e.g., AP/CM files)
        final_rows.append({
            "rank": i, "variant": variant, "H": H, "W": W,
            "activation": act, "lr": lr, "dropout": do, "flip_p": flip_p,
            "tag": tag, "started_at": datetime.now().isoformat(timespec="seconds")
        })

    # FINAL summary with clean columns, plus best metrics from GRID as reference
    final_csv = ART / f"final_summary_{variant}.csv"
    # map grid by (act, lr, dropout)
    grid_map = {(str(x.get("activation")), str(x.get("lr")), str(x.get("dropout"))): x for x in rows}
    with open(final_csv, "w", newline="") as f:
        fns = ["rank","variant","H","W","activation","lr","dropout","flip_p",
               "grid_best_val_acc","grid_best_val_loss","duration_sec_est","tag"]
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        dur = time.time() - start_all
        for r in final_rows:
            key = (r["activation"], r["lr"], r["dropout"])
            src = grid_map.get(key, {})
            w.writerow({
                "rank": r["rank"], "variant": r["variant"], "H": r["H"], "W": r["W"],
                "activation": r["activation"], "lr": r["lr"], "dropout": r["dropout"], "flip_p": r["flip_p"],
                "grid_best_val_acc": src.get("best_val_acc"), "grid_best_val_loss": src.get("best_val_loss"),
                "duration_sec_est": round(dur, 1), "tag": r["tag"],
            })
    print(f"WROTE Final summary -> {final_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True,
                    help="Which model variants to run (e.g., fc gap torch_fc torch_gap)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--topk", type=int, default=5, help="How many finalists to re-run in the FINAL stage")
    ap.add_argument("--flip_p", type=float, default=0.3)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    args = ap.parse_args()

    for v in args.variants:
        stage_banner(f"RUN PIPELINE — variant={v}")
        run_pipeline(variant=v, epochs=args.epochs, patience=args.patience,
                     topk=args.topk, flip_p=args.flip_p,
                     height=args.height, width=args.width)

if __name__ == "__main__":
    main()
