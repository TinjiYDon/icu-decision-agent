"""Generate verification heatmap PNG from the full L4 predict_grud pipeline."""
import sys, os
sys.path.insert(0, r"c:\Users\lewis\Desktop\icu-decision-agent-main (5)\icu-decision-agent-main")
os.chdir(r"c:\Users\lewis\Desktop\icu-decision-agent-main (5)\icu-decision-agent-main")

import numpy as np
from application.predict_grud import predict_grud
from presentation.ui.charts import fig_dual_encoded_heatmap

stay_id = 30000153
result = predict_grud(stay_id)
print(f"status={result['status']}, risk={result['risk_score']}")

gru = result["gru_attribution"]
vals = gru["per_timestep_values"]
attrs = gru["per_timestep_attribution"]
feat_names = gru["feature_names"]
time_labels = gru["time_labels"]
print(f"time_labels: {time_labels}")
print(f"top_factors (shap normalized):")
for f in result["top_factors"][:5]:
    print(f"  {f['feature']}: shap={f['shap']:.3f}")

fig = fig_dual_encoded_heatmap(vals, attrs, feat_names, time_labels)
OUT = r"c:\Users\lewis\Desktop\heatmap_final.png"
fig.write_image(OUT, width=1400, height=700, scale=2)
print(f"Saved: {OUT}")
