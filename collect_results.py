"""
collect_results.py
==================
تمام JSON های ablation_results/ را می‌خواند و نتایج همه مدل‌ها را در یک
فایل Excel با فرمت‌بندی مناسب برای پایان‌نامه ذخیره می‌کند.

اجرا:
    python collect_results.py
"""

import os, json, pathlib
from datetime import datetime

try:
    import openpyxl
    from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                                  numbers)
    from openpyxl.utils import get_column_letter
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "openpyxl", "--break-system-packages", "-q"])
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

# ─── مسیرها ───────────────────────────────────────────────────────────────
BASE        = pathlib.Path(__file__).parent
RESULTS_DIR = BASE / "ablation_results"
OUT_PATH    = BASE / f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

# ─── رنگ‌ها ────────────────────────────────────────────────────────────────
C_HEADER    = "2E4057"   # dark blue
C_SUBHEAD   = "457B9D"   # medium blue
C_BEST      = "D4EDDA"   # light green (best value)
C_ALT       = "F0F4F8"   # alternate row light blue
C_WHITE     = "FFFFFF"
C_TITLE     = "1A1A2E"

def _side(style="thin"): return Side(style=style)
def _border(): return Border(left=_side(), right=_side(), top=_side(), bottom=_side())

def _hdr(ws, row, col, text, bg=C_HEADER, fg="FFFFFF", bold=True,
         align="center", colspan=1):
    cell = ws.cell(row=row, column=col, value=text)
    cell.font      = Font(bold=bold, color=fg, size=11)
    cell.fill      = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
    cell.border    = _border()
    if colspan > 1:
        ws.merge_cells(start_row=row, start_column=col,
                       end_row=row, end_column=col + colspan - 1)
    return cell

def _val(ws, row, col, value, bg=C_WHITE, bold=False, fmt="0.0000"):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font      = Font(bold=bold, size=10)
    cell.fill      = PatternFill("solid", fgColor=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.border    = _border()
    if isinstance(value, float) and fmt:
        cell.number_format = fmt
    return cell

def _auto_width(ws, extra=4):
    for col in ws.columns:
        max_len = 0
        for cell in col:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[get_column_letter(col[0].column)].width = \
            min(max_len + extra, 40)


# ─── خواندن JSON ها ────────────────────────────────────────────────────────
def load_all_results():
    """تمام JSON ها را می‌خواند و نتایج را دسته‌بندی می‌کند."""
    model_rows   = []   # [{model, dataset, split, metrics, source}]
    ablation_rows = []
    baseline_rows = []

    json_files = sorted(RESULTS_DIR.glob("*.json"))

    for fp in json_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        name = fp.stem

        # ── test_results ──────────────────────────────────────────
        if name.startswith("test_results_"):
            ts = name.replace("test_results_", "")
            # تعیین مدل از timestamp (نزدیک‌ترین model file)
            if ts <= "20260507_150000":
                model_name = "ProtoNet"
                dataset    = "Japan v2"
            elif ts <= "20260507_200000":
                model_name = "ProtoNet"
                dataset    = "Asia v1"
            else:
                model_name = "IMETA-GNN"
                dataset    = "Japan v2 (alt)"
            model_rows.append({
                "model": model_name, "dataset": dataset,
                "split": "Test", "metrics": data.get("test_metrics", {}),
                "val_ndcg10": round(data.get("val_ndcg", 0), 4),
                "source": fp.name,
            })

        # ── melu_results ──────────────────────────────────────────
        elif name.startswith("melu_results_"):
            ds = data.get("dataset", "")
            ds_short = ds.split("/")[-1] if "/" in ds else (
                       ds.split("\\")[-1] if "\\" in ds else "Japan v2")
            model_rows.append({
                "model": "MeLU", "dataset": ds_short,
                "split": "Test", "metrics": data.get("test_metrics", {}),
                "val_ndcg10": round(
                    data.get("summary", {}).get("best_val_ndcg10",
                    data.get("val_metrics", {}).get("NDCG@10", 0)), 4),
                "source": fp.name,
            })

        # ── imetagnn_poi_social_results ───────────────────────────
        elif name.startswith("imetagnn_poi_social_results_"):
            ds = data.get("dataset", "")
            ds_short = ds.split("/")[-1] if "/" in ds else (
                       ds.split("\\")[-1] if "\\" in ds else "")
            if not ds_short:
                ds_short = "Japan v2" if "japan" in ds.lower() else (
                           "Asia v1"  if "asia"  in ds.lower() else "Unknown")
            model_rows.append({
                "model": "IMETA-GNN+POI+Social",
                "dataset": ds_short,
                "split": "Test",
                "metrics": data.get("test_metrics", {}),
                "val_ndcg10": round(
                    data.get("summary", {}).get("best_val_ndcg10",
                    data.get("val_metrics", {}).get("NDCG@10", 0)), 4),
                "source": fp.name,
            })

        # ── imetagnn_results ──────────────────────────────────────
        elif name.startswith("imetagnn_results_"):
            ds = data.get("dataset", "")
            ds_short = ds.split("/")[-1] if "/" in ds else (
                       ds.split("\\")[-1] if "\\" in ds else "")
            if not ds_short:
                # از checkpoint_dir حدس بزن
                ts_num = name.replace("imetagnn_results_", "")
                ds_short = "Asia v1" if ts_num >= "20260507_230000" else "Japan v2"
            model_rows.append({
                "model": "IMETA-GNN",
                "dataset": ds_short,
                "split": "Test",
                "metrics": data.get("test_metrics", {}),
                "val_ndcg10": round(
                    data.get("summary", {}).get("best_val_ndcg10",
                    data.get("val_metrics", {}).get("NDCG@10", 0)), 4),
                "source": fp.name,
            })

        # ── ablation_results ──────────────────────────────────────
        elif name.startswith("ablation_results_"):
            if isinstance(data, list):
                for item in data:
                    ablation_rows.append({
                        "variant": item.get("variant", "?"),
                        "best_epoch": item.get("best_epoch", "?"),
                        "metrics": item.get("best_metrics", {}),
                        "source": fp.name,
                    })

        # ── baseline_comparison ───────────────────────────────────
        elif name.startswith("baseline_comparison_"):
            if isinstance(data, list):
                for item in data:
                    baseline_rows.append({
                        "name": item.get("name", "?"),
                        "n_eval": item.get("n_eval", "?"),
                        "metrics": item.get("metrics", {}),
                        "source": fp.name,
                    })

    return model_rows, ablation_rows, baseline_rows


# ─── Sheet 1: مقایسه اصلی مدل‌ها ──────────────────────────────────────────
def write_model_sheet(wb, model_rows):
    ws = wb.create_sheet("مقایسه مدل‌ها")
    ws.sheet_view.rightToLeft = False

    # عنوان
    ws.row_dimensions[1].height = 30
    _hdr(ws, 1, 1,
         "نتایج نهایی مدل‌ها — پایان‌نامه کارشناسی ارشد",
         bg=C_TITLE, colspan=9)

    # سرستون
    ws.row_dimensions[2].height = 20
    headers = ["مدل", "دیتاست", "Precision@10", "Recall@10",
               "NDCG@10", "NDCG@20", "Recall@20", "Precision@20", "فایل منبع"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 2, c, h, bg=C_HEADER)

    # پیدا کردن بهترین مقدار برای هر معیار
    metrics_keys = ["Precision@10", "Recall@10", "NDCG@10",
                    "NDCG@20", "Recall@20", "Precision@20"]

    # حذف تکراری‌ها: برای هر (model, dataset) آخرین نتیجه را نگه می‌داریم
    dedup = {}
    for r in model_rows:
        key = (r["model"], r["dataset"])
        dedup[key] = r
    rows = list(dedup.values())
    rows.sort(key=lambda x: (x["dataset"], x["model"]))

    # best per metric
    best = {}
    for mk in metrics_keys:
        vals = [r["metrics"].get(mk, 0) for r in rows if r["metrics"].get(mk)]
        best[mk] = max(vals) if vals else -1

    for i, row in enumerate(rows, 3):
        bg = C_ALT if i % 2 == 0 else C_WHITE
        ws.row_dimensions[i].height = 18
        _val(ws, i, 1, row["model"],     bg=bg, bold=True, fmt=None)
        _val(ws, i, 2, row["dataset"],   bg=bg, fmt=None)
        for ci, mk in enumerate(metrics_keys, 3):
            v = row["metrics"].get(mk, "—")
            cell_bg = C_BEST if isinstance(v, float) and abs(v - best.get(mk, -1)) < 1e-8 else bg
            _val(ws, i, ci, v, bg=cell_bg)
        _val(ws, i, 9, row["source"], bg=bg, fmt=None)

    # اضافه کردن ProtoNet با مقادیر از log
    # (اگه test_results فایل جداگانه هستند اینجا هستند)

    _auto_width(ws)
    ws.freeze_panes = "A3"
    return ws


# ─── Sheet 2: Ablation Study ───────────────────────────────────────────────
def write_ablation_sheet(wb, ablation_rows):
    ws = wb.create_sheet("Ablation Study")
    ws.row_dimensions[1].height = 28

    _hdr(ws, 1, 1, "Ablation Study — ProtoNet (Japan v2)",
         bg=C_TITLE, colspan=7)
    headers = ["Variant", "Best Epoch",
               "Precision@10", "Recall@10", "NDCG@10", "NDCG@20", "Recall@20"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 2, c, h, bg=C_SUBHEAD)

    # آخرین ablation file (بیشترین timestamp)
    if ablation_rows:
        latest_src = sorted(set(r["source"] for r in ablation_rows))[-1]
        rows = [r for r in ablation_rows if r["source"] == latest_src]
    else:
        rows = ablation_rows

    for i, row in enumerate(rows, 3):
        bg = C_ALT if i % 2 == 0 else C_WHITE
        _val(ws, i, 1, row["variant"], bg=bg, bold=True, fmt=None)
        _val(ws, i, 2, row["best_epoch"], bg=bg, fmt=None)
        for ci, mk in enumerate(["Precision@10","Recall@10","NDCG@10","NDCG@20","Recall@20"], 3):
            _val(ws, i, ci, row["metrics"].get(mk, "—"), bg=bg)

    _auto_width(ws)
    ws.freeze_panes = "A3"


# ─── Sheet 3: Baselines ───────────────────────────────────────────────────
def write_baseline_sheet(wb, baseline_rows):
    ws = wb.create_sheet("Baselines")
    _hdr(ws, 1, 1, "Baseline Comparison (Japan v2, 500 val users)",
         bg=C_TITLE, colspan=7)
    headers = ["روش", "# کاربر",
               "Precision@10", "Recall@10", "NDCG@10", "NDCG@20", "Recall@20"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 2, c, h, bg=C_SUBHEAD)

    # آخرین baseline file
    if baseline_rows:
        latest_src = sorted(set(r["source"] for r in baseline_rows))[-1]
        rows = [r for r in baseline_rows if r["source"] == latest_src]
    else:
        rows = baseline_rows

    for i, row in enumerate(rows, 3):
        bg = C_ALT if i % 2 == 0 else C_WHITE
        _val(ws, i, 1, row["name"],   bg=bg, bold=True, fmt=None)
        _val(ws, i, 2, row["n_eval"], bg=bg, fmt=None)
        for ci, mk in enumerate(["Precision@10","Recall@10","NDCG@10","NDCG@20","Recall@20"], 3):
            _val(ws, i, ci, row["metrics"].get(mk, "—"), bg=bg)

    # اضافه کردن ProtoNet برای مقایسه
    i = len(rows) + 3
    _hdr(ws, i, 1, "ProtoNet (ours)", bg="1D6A4D", colspan=1)
    for ci, v in enumerate([0.3173, 0.2725, 0.3174, 0.3570, 0.3812], 2):
        cell = ws.cell(row=i, column=ci + 1, value=v)
        cell.font  = Font(bold=True, color="FFFFFF", size=10)
        cell.fill  = PatternFill("solid", fgColor="1D6A4D")
        cell.alignment = Alignment(horizontal="center")
        cell.number_format = "0.0000"
        cell.border = _border()

    _auto_width(ws)
    ws.freeze_panes = "A3"


# ─── Sheet 4: خلاصه 4 معیار ───────────────────────────────────────────────
def write_summary_sheet(wb, model_rows):
    """Sheet اول و اصلی: جدول ۴ معیار اصلی برای همه مدل‌ها"""
    ws = wb.create_sheet("خلاصه ۴ معیار", 0)   # اول بگذار

    ws.row_dimensions[1].height = 35
    ws.row_dimensions[2].height = 22

    title = "مقایسه نهایی — ۴ معیار اصلی (Test Set)"
    _hdr(ws, 1, 1, title, bg=C_TITLE, colspan=6)

    headers = ["مدل / Baseline", "Dataset",
               "NDCG@10 ↑", "NDCG@20 ↑", "Recall@10 ↑", "Recall@20 ↑"]
    for c, h in enumerate(headers, 1):
        _hdr(ws, 2, c, h, bg=C_HEADER)

    # deduplicate
    dedup = {}
    for r in model_rows:
        key = (r["model"], r["dataset"])
        dedup[key] = r
    rows = list(dedup.values())
    rows.sort(key=lambda x: (x["dataset"], x["model"]))

    best = {}
    for mk in ["NDCG@10", "NDCG@20", "Recall@10", "Recall@20"]:
        vals = [r["metrics"].get(mk, 0) for r in rows]
        best[mk] = max(vals) if vals else -1

    for i, row in enumerate(rows, 3):
        bg = C_ALT if i % 2 == 0 else C_WHITE
        ws.row_dimensions[i].height = 18
        _val(ws, i, 1, row["model"],   bg=bg, bold=True, fmt=None)
        _val(ws, i, 2, row["dataset"], bg=bg, fmt=None)
        for ci, mk in enumerate(["NDCG@10","NDCG@20","Recall@10","Recall@20"], 3):
            v = row["metrics"].get(mk, "—")
            cell_bg = C_BEST if isinstance(v, float) and abs(v - best.get(mk, -1)) < 1e-8 else bg
            cell = _val(ws, i, ci, v, bg=cell_bg)
            if isinstance(v, float) and abs(v - best.get(mk, -1)) < 1e-8:
                cell.font = Font(bold=True, size=11, color="155724")

    # پانوشت
    note_row = len(rows) + 4
    ws.cell(row=note_row, column=1,
            value="* سلول‌های سبز = بهترین مقدار در هر معیار")
    ws.cell(row=note_row, column=1).font = Font(italic=True, size=9, color="666666")

    _auto_width(ws)
    ws.freeze_panes = "A3"


# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    print("📂 خواندن نتایج از:", RESULTS_DIR)
    model_rows, ablation_rows, baseline_rows = load_all_results()

    print(f"  ✓ {len(model_rows)} رکورد مدل یافت شد")
    print(f"  ✓ {len(ablation_rows)} variant ablation یافت شد")
    print(f"  ✓ {len(baseline_rows)} baseline یافت شد")

    wb = openpyxl.Workbook()
    # sheet پیش‌فرض را حذف کن
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    write_summary_sheet(wb, model_rows)
    write_model_sheet(wb, model_rows)
    write_ablation_sheet(wb, ablation_rows)
    write_baseline_sheet(wb, baseline_rows)

    wb.save(str(OUT_PATH))
    print(f"\n✅ Excel ذخیره شد: {OUT_PATH}")
    return str(OUT_PATH)


if __name__ == "__main__":
    main()
