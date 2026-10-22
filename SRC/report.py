import base64
from pathlib import Path
import markdown


def build_html(report_md, plots_dir="outputs/plots"):
    """Turn the Markdown report into one self-contained HTML file with plots embedded."""
    body = markdown.markdown(report_md, extensions=["tables"])
    figures = ""
    for p in sorted(Path(plots_dir).glob("*.png")):
        data = base64.b64encode(p.read_bytes()).decode()
        figures += (f'<figure><img src="data:image/png;base64,{data}">'
                    f"<figcaption>{p.stem}</figcaption></figure>\n")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EDA report</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 900px; margin: 40px auto;
          padding: 0 20px; line-height: 1.6; color: #222; }}
  h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
  figure {{ margin: 20px 0; }} img {{ max-width: 100%; }}
  figcaption {{ color: #666; font-size: 0.9em; }}
</style></head>
<body>{body}<h2>Figures</h2>{figures}</body></html>"""