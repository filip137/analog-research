"""One HTML selector for CIFAR dataset, device, accuracy and teacher KL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import plot_cifar_best_method as best

base = best.base

PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CIFAR recovery explorer</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;color:#21303b;max-width:1540px;margin:25px auto;padding:0 22px}
h1{font-size:28px;margin-bottom:5px}p{max-width:1230px}a{color:#2166ac}.controls{display:flex;flex-wrap:wrap;gap:22px;padding:18px;background:#edf2f5;border-radius:8px;margin:24px 0 16px}
label{display:flex;flex-direction:column;gap:5px;font-size:13px}select{font:16px system-ui;padding:9px 14px;background:white;border:1px solid #afbdc7;border-radius:5px;min-width:150px}select:disabled{opacity:.5}
#plot{width:100%;height:auto;border:1px solid #e6ebef}#status{font-weight:600}#error{color:#a63030;padding:12px;background:#fff0f0}details{margin:22px 0}summary{cursor:pointer;font-weight:600}.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:12px;width:100%;margin-top:15px}td,th{padding:9px;border-bottom:1px solid #d8e0e5;text-align:right}th{background:#edf2f5}td:nth-child(-n+4){text-align:left}.muted{color:#5d6972;font-size:14px}
@media print{.controls,.downloads,details{display:none}body{margin:0}#plot{border:0}}
</style></head><body>
<h1>CIFAR recovery explorer</h1>
<p>Choose the dataset, device and metric. Each view shows both analog depths and all three fault types, comparing the best calibrated method with on-chip recovery.</p>
<div class="controls">
<label>Dataset<select id="dataset"><option value="cifar100">CIFAR-100</option><option value="cifar10">CIFAR-10</option></select></label>
<label>Device<select id="device"><option value="om">OM</option><option value="pcm">PCM</option></select></label>
<label>Plot<select id="metric"><option value="teacher_kl">KL divergence</option><option value="accuracy_percent">Accuracy</option></select></label>
<label>KL axis<select id="scale"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></label>
</div>
<p id="status" aria-live="polite"></p>
<p class="downloads"><a id="jpg">Download JPG</a> · <a id="png">PNG</a> · <a id="svg">SVG</a> · <a id="pdf">PDF</a> · <a id="csv">Data CSV</a></p>
<p id="error" hidden>The plot could not load. Keep this HTML file beside its accompanying assets folder.</p>
<a id="fullsize" target="_blank" rel="noopener"><img id="plot" width="2310" height="1320" alt="Selected CIFAR best-method and recovery comparison"></a>
<p><strong>Colour identifies the training method.</strong> Open circles show the source with the lowest mean KL after calibration. Filled squares show recovery of that same source. A diamond marks a different source that achieves a lower KL after recovery.</p>
<p id="metric-note"></p>
<p class="muted">Both views use the same sources selected by lowest observed mean KL. Accuracy is not re-ranked. Means and sample SD describe three modeled arrays; rates are 0%, 1%, 2%, 3%, and 5%. Both adaptation arms use five epochs from their initial deployment. Click the figure to open the vector version.</p>
<details><summary>Exact values for this dataset and device</summary><div class="scroll" id="table"></div></details>
<details><summary>How to interpret the comparison</summary>
<p>Gains below the axes compare calibration-only and weight-recovery-plus-calibration outcomes of the same source. Positive means improvement. Relative KL gain is computed per array and then averaged; a small baseline KL can produce a large negative percentage. A different-source diamond compares another complete pipeline. Its difference from the calibrated winner can include a change of offline training method.</p>
<p>All source choices summarize the frozen test results retrospectively. They are not a validated deployment-selection policy. Original convergence flags remain in the data. The classifier is analog along with the stated four/eight convolutions. 0% retains modeled programming error; 4% was not measured. The frozen historical OM closed-loop recovery settings are used; later LR and open-loop studies are separate.</p>
</details>
<script id="records" type="application/json">DATA_JSON</script>
<script>
'use strict';
const records=JSON.parse(document.getElementById('records').textContent);
const assets=ASSETS_JSON;
const sourceNames={digital:'No HWA',standard_hwa:'Normal HWA',noise_hwa:'Noisy HWA',cdt:'Corruption-aware HWA'};
const faultNames={open:'Stuck-low / open',random:'Random-stuck',gmax:'Stuck-high'};
const fields=['dataset','device','metric','scale'];
const controls=Object.fromEntries(fields.map(k=>[k,document.getElementById(k)]));
function values(){return Object.fromEntries(fields.map(k=>[k,controls[k].value]));}
function pair(r,s){return Number(r[s+'_teacher_kl_mean']).toPrecision(4)+' / '+Number(r[s+'_accuracy_percent_mean']).toFixed(2)+'%';}
function signed(v,d=4){v=Number(v);return (v>=0?'+':'')+v.toFixed(d);}
function update(){
 const v=values(),accuracy=v.metric==='accuracy_percent';controls.scale.disabled=accuracy;
 const view=accuracy?'accuracy':'kl_'+v.scale,stem=v.dataset+'_'+v.device+'_best_method_'+view;
 const datasetLabel=v.dataset==='cifar100'?'CIFAR-100':'CIFAR-10';
 document.getElementById('status').textContent=datasetLabel+' · '+v.device.toUpperCase()+' · '+(accuracy?'Accuracy':'Teacher KL ('+v.scale+' scale)');
 for(const ext of ['jpg','png','svg','pdf']){const a=document.getElementById(ext);a.href=assets+'/'+stem+'.'+ext;a.download=stem+'.'+ext;}
 document.getElementById('csv').href=assets+'/best_method_summary.csv';
 document.getElementById('fullsize').href=assets+'/'+stem+'.svg';
 document.getElementById('error').hidden=true;
 const img=document.getElementById('plot');img.src=assets+'/'+stem+'.png';
 img.alt=datasetLabel+' '+v.device.toUpperCase()+' '+(accuracy?'accuracy':'teacher KL')+' for best calibrated and recovered methods';
 document.getElementById('metric-note').textContent=accuracy?'Higher accuracy is better. Numbers below each rate show the matched accuracy gain in percentage points. The dotted horizontal reference is the clean digital teacher.':'Lower KL is better. Numbers below each rate show the matched percentage KL reduction. KL is measured from the original digital teacher to the deployed model, in nats at temperature 1.';
 const selected=records.filter(r=>r.dataset===v.dataset&&r.backend===v.device);
 let t='<table><thead><tr><th>Analog conv.</th><th>Fault</th><th>Rate</th><th>Best calibrated source</th><th>Cal. KL / accuracy</th><th>Same-source recovery KL / accuracy</th><th>ΔKL (nats)</th><th>Δ accuracy (pp)</th><th>Best recovered source</th><th>Best recovered KL / accuracy</th></tr></thead><tbody>';
 for(const r of selected)t+='<tr><td>'+r.depth+'</td><td>'+faultNames[r.fault_kind]+'</td><td>'+r.fault_rate_percent+'%</td><td>'+sourceNames[r.calibration_family]+'</td><td>'+pair(r,'calibration')+'</td><td>'+pair(r,'matched_recovery')+'</td><td>'+signed(r.matched_kl_gain_nats_mean)+'</td><td>'+signed(r.matched_accuracy_gain_pp_mean,2)+'</td><td>'+sourceNames[r.recovery_family]+'</td><td>'+pair(r,'best_recovery')+'</td></tr>';
 document.getElementById('table').innerHTML=t+'</tbody></table>';
 document.body.dataset.selection=stem;document.body.dataset.rowCount=String(selected.length);
 const hash=new URLSearchParams(v).toString();if(location.hash.slice(1)!==hash)history.replaceState(null,'','#'+hash);
}
function readHash(){const h=new URLSearchParams(location.hash.slice(1));for(const k of fields){const v=h.get(k);if([...controls[k].options].some(o=>o.value===v))controls[k].value=v;}update();}
for(const c of Object.values(controls))c.addEventListener('change',update);
document.getElementById('plot').addEventListener('error',()=>{document.getElementById('error').hidden=false;});
window.addEventListener('hashchange',readHash);readHash();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    inputs = args.input_dir or root / "results/cifar-sweep-analysis"
    audit = args.source_audit or root / "artifacts/cifar_recovery_full_results_20260920/source_fit_selection.csv"
    out = (args.output or root / "artifacts/cifar_kl_fault_curves_20260922/best_method/explorer.html").resolve()
    assets = out.with_name(out.stem + "_assets")
    assets.mkdir(parents=True, exist_ok=True)
    paths = {n: inputs / n for n in ("summary.csv", "per_array.csv", "clean_sources.csv")}
    index, raw, clean, *_ = base.load_data(paths)
    records, paired, candidates = best.summarize(index, raw, base.read_csv(audit))
    for name, rows in (("best_method_summary.csv", records), ("paired_arrays.csv", paired), ("candidate_endpoints.csv", candidates)):
        base.write_csv(assets / name, rows)
    base.plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    figures = []
    for dataset in base.DATASETS:
        for device in ("om", "pcm"):
            for view in best.VIEWS:
                fig = best.figure(dataset, view, records, clean, device=device)
                stem = f"{dataset}_{device}_best_method_{view}"
                fig.savefig(assets / (stem + ".png"), dpi=150, facecolor="white")
                fig.savefig(assets / (stem + ".jpg"), dpi=200, facecolor="white", pil_kwargs={"quality": 95, "subsampling": 0})
                fig.savefig(assets / (stem + ".svg"), facecolor="white")
                fig.savefig(assets / (stem + ".pdf"), facecolor="white")
                base.plt.close(fig)
                figures.append(stem)
                print(stem, flush=True)
    out.write_text(PAGE.replace("DATA_JSON", json.dumps(records)).replace("ASSETS_JSON", json.dumps(assets.name)))
    for script in (Path(__file__), Path(best.__file__), Path(base.__file__)):
        if script.resolve() != (assets / script.name).resolve():
            shutil.copyfile(script, assets / script.name)
    paths["source_fit_selection.csv"] = audit
    validation = dict(status="passed", html=str(out), figures=figures, plotted_conditions=len(records),
        selection_metric="Lowest observed mean teacher KL; unchanged between accuracy and KL views",
        input_files={n: dict(path=str(p.resolve()), sha256=base.sha256(p)) for n, p in paths.items()},
        script_hashes={p.name: base.sha256(p) for p in (Path(__file__), Path(best.__file__), Path(base.__file__))})
    (assets / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (assets / "README.md").write_text('''# Unified CIFAR HTML explorer

Open `../explorer.html` and use the dataset, device, plot and KL-scale selectors.
Default: CIFAR-100, OM, logarithmic teacher KL. Accuracy uses the same KL-selected
models. Each figure contains two analog depths and three fault types. All data,
source choices and paired-gain definitions match the previous best-method export.
The HTML loads its PNG/SVG/JPG/PDF files from this folder. Keep the HTML beside
this assets folder when moving it. No server or internet connection is required.

Reproduce from the `best_method` folder of the complete bundle:

```sh
python explorer_assets/build_cifar_explorer.py --input-dir ../inputs --source-audit ../inputs/source_fit_selection.csv --output explorer_rebuilt.html
```
''')
    print(json.dumps({"html": str(out), "views": len(figures)}))


if __name__ == "__main__":
    main()
