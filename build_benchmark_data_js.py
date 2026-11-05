import json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path("d:/neural-bates-dml")

# 1. SPY Snapshot
df = pd.read_csv(REPO_ROOT / "data" / "eval" / "spy_options_snapshot.csv")
spy_records = df[[
    'expiry', 'dte', 'T', 'spot', 'strike', 'moneyness', 'log_moneyness', 
    'option_type', 'mid', 'market_iv', 'r', 'q'
]].to_dict(orient='records')

# 2. Calibration benchmark
with open(REPO_ROOT / "results" / "spy_calibration_benchmark.json") as f:
    calib_bench = json.load(f)

# 3. Ablation study
with open(REPO_ROOT / "results" / "ablation_study.json") as f:
    ablation = json.load(f)

output_path = REPO_ROOT / "benchmark_data.js"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("// Auto-generated empirical benchmark data for Neural Bates DML Options Workbench\n")
    f.write("var SPY_SNAPSHOT = " + json.dumps(spy_records) + ";\n")
    f.write("var CALIBRATION_BENCHMARK = " + json.dumps(calib_bench) + ";\n")
    f.write("var ABLATION_STUDY = " + json.dumps(ablation) + ";\n")
    f.write("if (typeof module !== 'undefined') { module.exports = { SPY_SNAPSHOT, CALIBRATION_BENCHMARK, ABLATION_STUDY }; }\n")
    f.write("if (typeof window !== 'undefined') { window.SPY_SNAPSHOT = SPY_SNAPSHOT; window.CALIBRATION_BENCHMARK = CALIBRATION_BENCHMARK; window.ABLATION_STUDY = ABLATION_STUDY; }\n")

import os
print("Wrote benchmark_data.js:", os.path.getsize(output_path) / 1024, "KB")
