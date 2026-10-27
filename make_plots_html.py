import base64
import html
import os

plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output", "plots")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output", "plots.html")

cards = []
for f in sorted(os.listdir(plots_dir)):
    if f.endswith(".png"):
        b64 = base64.b64encode(open(os.path.join(plots_dir, f), "rb").read()).decode()
        cards.append('<h2>{}</h2><img src="data:image/png;base64,{}" style="max-width:100%;border:1px solid #1C2A38;border-radius:6px;">'.format(html.escape(f), b64))

page = (
    '<!DOCTYPE html><html><head><meta charset="utf-8">'
    '<title>DermAI - Evaluation Charts</title>'
    '<style>body{background:#070B0F;color:#E8EDF2;font-family:Inter,sans-serif;max-width:1100px;margin:0 auto;padding:24px}'
    'h1{color:#14B8A6;font-size:20px}h2{color:#8A9BB0;font-size:13px;margin-top:28px;font-family:monospace}</style></head>'
    '<body><h1>DermAI Performance &amp; Confidence Charts</h1>'
    '<p style="color:#8A9BB0;font-size:13px">Generated from model_weights.pth on the DermNet test split '
    '(4002 images, empirically aligned label mapping).</p>'
    + "".join(cards)
    + "</body></html>"
)

with open(out, "w") as fp:
    fp.write(page)
print("wrote", out, len(page), "bytes")