
import argparse, sys, subprocess, json, csv, os, time
from pathlib import Path

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, check=True)
    return r

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def pick_topk(csv_path, k=5):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row["best_val_acc"] = float(row["best_val_acc"])
            rows.append(row)
    rows.sort(key=lambda r: r["best_val_acc"], reverse=True)
    return rows[:k]

def stage_banner(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def run_pipeline(variant: str, epochs: int, patience: int, flip_p: float,
                 pretrained: bool=False, enforce_7x7: bool=False, batch_size: int=32):
    # ---- Stage 1: SIZE ----
    stage_banner(f"[{variant.upper()}] STAGE 1 — SIZE")
    cmd = [sys.executable, "tuning.py",
           "--variant", variant, "--stage", "size",
           "--epochs", str(epochs), "--patience", str(patience)]
    if pretrained: cmd.append("--pretrained")
    if enforce_7x7: cmd.append("--enforce_7x7")
    run(cmd)
    best_size = read_json(ART / f"best_{variant}_size.json")
    H, W = int(best_size["H"]), int(best_size["W"])
    print(f"STAGE 1 COMPLETE! Winner: H={H}, W={W}", flush=True)

    # ---- Stage 2: GRAYSCALE ----
    stage_banner(f"[{variant.upper()}] STAGE 2 — GRAYSCALE")
    cmd = [sys.executable, "tuning.py",
           "--variant", variant, "--stage", "grayscale",
           "--epochs", str(epochs), "--patience", str(patience),
           "--height", str(H), "--width", str(W)]
    if pretrained: cmd.append("--pretrained")
    if enforce_7x7: cmd.append("--enforce_7x7")
    run(cmd)
    best_gray = read_json(ART / f"best_{variant}_grayscale.json")
    gray = bool(best_gray["grayscale"])
    print(f"STAGE 2 COMPLETE! Winner: grayscale={gray}", flush=True)

    # ---- Stage 3: ACT × LR × DROPOUT ----
    stage_banner(f"[{variant.upper()}] STAGE 3 — ACTIVATION × LR × DROPOUT")
    cmd = [sys.executable, "tuning.py",
           "--variant", variant, "--stage", "act_lr_dropout",
           "--epochs", str(epochs), "--patience", str(patience),
           "--height", str(H), "--width", str(W)]
    if gray: cmd.append("--grayscale")
    if pretrained: cmd.append("--pretrained")
    if enforce_7x7: cmd.append("--enforce_7x7")
    run(cmd)

    grid_csv = ART / f"tuning_{variant}_act_lr_dropout.csv"
    top5 = pick_topk(grid_csv, 5)
    print("STAGE 3 COMPLETE! Top‑5 (by best_val_acc):")
    for i, r in enumerate(top5, 1):
        print(f"  {i}. act={r['activation']}, lr={r['lr']}, drop={r.get('dropout')}, best_val_acc={r['best_val_acc']:.4f}")

    # ---- Stage 4: AUGMENT (Top‑5) + FINAL EVAL via main.py ----
    stage_banner(f"[{variant.upper()}] STAGE 4 — AUGMENT & FINAL REPORTS (Top‑5)")
    top5_summary_rows = []
    for i, r in enumerate(top5, 1):
        act = r["activation"]
        lr  = r["lr"]
        drop = r.get("dropout")
        print(f"\n== Augment run {i}/5: act={act} lr={lr} drop={drop} flip_p={flip_p} ==")
        cmd = [sys.executable, "tuning.py",
               "--variant", variant, "--stage", "augment",
               "--epochs", str(epochs), "--patience", str(patience),
               "--height", str(H), "--width", str(W),
               "--activation", act, "--lr", str(lr),
               "--dropout", str(drop),]
        if gray: cmd.append("--grayscale")
        if pretrained: cmd.append("--pretrained")
        if enforce_7x7: cmd.append("--enforce_7x7")
        # fix flip_p via env passthrough (tuning.py augment stage fixes flip_p to 0.3 right now),
        # so we just leave it; or we can simply run main.py with desired flip afterwards.
        run(cmd)

        # Final report artifacts with main.py
        tag_variant = {
            "fc": "fc", "gap": "gap", "torch_fc": "torch_fc", "torch_gap": "torch_gap"
        }[variant]
        main_cmd = [sys.executable, "main.py",
                    "--variant", tag_variant,
                    "--epochs", str(epochs),
                    "--height", str(H), "--width", str(W),
                    "--activation", act,
                    "--lr", str(lr)]
        if drop not in (None, "", "None"):
            main_cmd += ["--dropout", str(drop)]
        if gray: main_cmd.append("--grayscale")
        # pass augmentation to final eval grid image
        if flip_p and float(flip_p) > 0:
            main_cmd += ["--flip_p", str(flip_p)]
        if pretrained and ("torch_" in variant):
            main_cmd.append("--pretrained")
        if enforce_7x7 and (variant == "gap"):
            main_cmd.append("--enforce_7x7")

        print(f"Running final report for Top‑{i} via main.py ...")
        run(main_cmd)

        # Collect to summary
        top5_summary_rows.append({
            "rank": i,
            "variant": variant,
            "H": H, "W": W, "grayscale": gray,
            "activation": act, "lr": lr, "dropout": drop,
        })

    # write summary CSV
    out_csv = ART / f"top5_summary_{variant}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(top5_summary_rows[0].keys()))
        w.writeheader(); w.writerows(top5_summary_rows)
    print(f"\nWROTE Top‑5 summary -> {out_csv}")

    # nice banner end
    print("\n" + "#"*80)
    print(f"PIPELINE COMPLETE for {variant.upper()} — see 'artifacts/' for CSVs, JPGs, and TensorBoard logs in 'runs/'")
    print("#"*80)

def parse_args():
    ap = argparse.ArgumentParser(description="Full pipeline runner (no PowerShell).")
    ap.add_argument("--variants", type=str, default="fc,gap,torch_fc",
                    help="Comma‑separated list: fc,gap,torch_fc,torch_gap")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--flip_p", type=float, default=0.3, help="Augmentation prob used in final main.py runs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet weights for torchvision variants")
    ap.add_argument("--enforce_7x7", action="store_true", help="Require 7x7 feature map for GAP variant")
    return ap.parse_args()

def main():
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    start_time = time.time()

    for v in variants:
        run_pipeline(
            variant=v,
            epochs=args.epochs,
            patience=args.patience,
            flip_p=args.flip_p,
            pretrained=args.pretrained,
            enforce_7x7=args.enforce_7x7,
            batch_size=args.batch_size,
        )

    mins = (time.time() - start_time) / 60.0
    print(f"\nAll requested variants finished in ~{mins:.1f} min")

if __name__ == "__main__":
    main()
