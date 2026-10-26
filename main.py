#!/usr/bin/env python3
"""Exploratory BJJ state-transition and semi-Markov analysis."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ijson

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
VICOS_URL = "https://data.vicos.si/datasets/JuiJuitsu/annotations.json"
FPS = 30.0


def normalize_state(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_vicos(path: Path) -> tuple[pd.DataFrame, dict]:
    """Convert ViCoS frame labels to one row per sequence/frame.

    The first two image-name digits identify a source sequence; the annotation's
    explicit frame field is preferred for time ordering. Conflicting duplicate
    labels are resolved by the most frequent label and counted in provenance.
    """
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    annotation_records = 0
    with path.open("rb") as f:
        for item in ijson.items(f, "item"):
            annotation_records += 1
            image = str(item.get("image", ""))
            if not image:
                continue
            seq = image[:2]
            try:
                frame = int(item.get("frame", image[-5:]))
            except (TypeError, ValueError):
                continue
            state = normalize_state(item.get("position", ""))
            if state:
                grouped[(seq, frame)].append(state)
    rows, conflicts = [], 0
    for (seq, frame), labels in sorted(grouped.items()):
        counts = Counter(labels)
        if len(counts) > 1:
            conflicts += 1
        rows.append({"sequence_id": seq, "frame": frame, "position": counts.most_common(1)[0][0]})
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    return pd.DataFrame(rows), {"source": "ViCoS BJJ Positions Dataset", "annotation_records": annotation_records, "unique_sequence_frames": len(rows), "duplicate_frame_label_conflicts": conflicts, "annotation_sha256":digest.hexdigest(), "fps_assumption": FPS}


def make_demo() -> tuple[pd.DataFrame, dict]:
    """Small synthetic sequence fixture for smoke/demo runs; not observed sport data."""
    rng = np.random.default_rng(20261026)
    states = ["standing", "closed_guard1", "half_guard1", "side_control2", "mount2", "back2"]
    p = np.array([
        [.60,.25,.05,.05,.03,.02], [.05,.38,.30,.10,.10,.07],
        [.02,.08,.42,.28,.12,.08], [.01,.02,.08,.48,.25,.16],
        [.01,.02,.03,.18,.48,.28], [.01,.01,.01,.04,.13,.81],
    ])
    p = p / p.sum(axis=1, keepdims=True)
    rows=[]
    for seq in range(6):
        state_i=0; frame=1
        for _ in range(350):
            hold=int(rng.integers(2,75)); state=states[state_i]
            for k in range(hold): rows.append({"sequence_id":f"D{seq:02d}","frame":frame+k,"position":state})
            frame += hold
            state_i=int(rng.choice(len(states),p=p[state_i]))
    return pd.DataFrame(rows), {"source":"SYNTHETIC DEMO ONLY", "seed":20261026, "fps_assumption":FPS}


def to_episodes(frames: pd.DataFrame, fps: float, max_gap_seconds: float) -> tuple[pd.DataFrame, int]:
    required={"sequence_id","frame","position"}
    if not required.issubset(frames.columns):
        raise ValueError(f"Input must contain columns: {', '.join(sorted(required))}")
    f=frames.copy()
    f["sequence_id"]=f["sequence_id"].astype(str)
    f["frame"]=pd.to_numeric(f["frame"],errors="coerce")
    f["position"]=f["position"].map(normalize_state)
    f=f.dropna(subset=["sequence_id","frame"])
    f["frame"]=f["frame"].astype(int)
    f=f[f["position"]!=""].drop_duplicates(["sequence_id","frame"],keep="first")
    f=f.sort_values(["sequence_id","frame"],kind="stable")
    max_gap=max(1,int(round(fps*max_gap_seconds)))
    eps=[]; gap_breaks=0
    for seq,g in f.groupby("sequence_id",sort=True):
        rec=g.to_dict("records")
        if not rec: continue
        start=rec[0]["frame"]; prev=rec[0]["frame"]; state=rec[0]["position"]
        first=True
        for row in rec[1:]:
            fr=row["frame"]; st=row["position"]; gap=fr-prev
            if gap>max_gap:
                eps.append({"sequence_id":seq,"position":state,"start_frame":start,"end_frame":prev,"next_position":"","duration_seconds":max(0,(prev-start)/fps),"event_observed":False,"left_censored":first,"gap_censored":True})
                gap_breaks+=1; start=fr; state=st; first=True
            elif st!=state:
                eps.append({"sequence_id":seq,"position":state,"start_frame":start,"end_frame":prev,"next_position":st,"duration_seconds":max(0,(fr-start)/fps),"event_observed":True,"left_censored":first,"gap_censored":False})
                start=fr; state=st; first=False
            prev=fr
        eps.append({"sequence_id":seq,"position":state,"start_frame":start,"end_frame":prev,"next_position":"","duration_seconds":max(0,(prev-start)/fps),"event_observed":False,"left_censored":first,"gap_censored":False})
    return pd.DataFrame(eps),gap_breaks


def km_table(episodes: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    # Left-censored first observations are excluded; right-censored terminal
    # observations remain in the risk set and are not counted as exits.
    data=episodes[~episodes["left_censored"]]
    for state,g in data.groupby("position",sort=True):
        times=np.sort(g["duration_seconds"].to_numpy(dtype=float))
        events=g["event_observed"].to_numpy(dtype=bool)
        surv=1.0
        for t in np.unique(times):
            at_risk=int(np.sum(times>=t)); d=int(np.sum((times==t)&events)); c=int(np.sum((times==t)&~events))
            if at_risk and d: surv*=1-d/at_risk
            rows.append({"position":state,"time_seconds":float(t),"at_risk":at_risk,"exits":d,"right_censored":c,"survival_probability":surv})
    return pd.DataFrame(rows)


def transition_matrix(episodes: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    observed=episodes[episodes["event_observed"] & episodes["next_position"].ne("")]
    counts=pd.crosstab(observed["position"],observed["next_position"],dropna=False)
    counts=counts.reindex(index=sorted(set(episodes.position)),columns=sorted(set(episodes.position)),fill_value=0)
    probs=counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    return counts,probs


def sequence_bootstrap(episodes: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    observed=episodes[episodes["event_observed"] & episodes["next_position"].ne("")]
    sequences=sorted(observed.sequence_id.unique())
    if len(sequences)<2 or reps<=0: return pd.DataFrame(columns=["from_position","to_position","p2_5","p97_5","bootstrap_reps"])
    states=sorted(set(observed.position)|set(observed.next_position))
    state_index={s:i for i,s in enumerate(states)}
    seq_mats={s:np.zeros((len(states),len(states)),dtype=np.int64) for s in sequences}
    for row in observed.itertuples(index=False):
        seq_mats[row.sequence_id][state_index[row.position],state_index[row.next_position]] += 1
    stack=np.stack([seq_mats[s] for s in sequences])
    pairs=sorted(set(zip(observed.position,observed.next_position)))
    rng=np.random.default_rng(seed); vals=defaultdict(list)
    for _ in range(reps):
        aggregate=stack[rng.integers(0,len(sequences),size=len(sequences))].sum(axis=0)
        rowtot=aggregate.sum(axis=1)
        for a,b in pairs:
            i,j=state_index[a],state_index[b]
            if rowtot[i] > 0:
                vals[(a,b)].append(float(aggregate[i,j]/rowtot[i]))
    return pd.DataFrame([{"from_position":a,"to_position":b,"p2_5":float(np.quantile(v,.025)),"p97_5":float(np.quantile(v,.975)),"bootstrap_reps":reps} for (a,b),v in vals.items()])


def save_outputs(frames: pd.DataFrame, provenance: dict, fps: float, max_gap: float, bootstrap: int) -> dict:
    OUT.mkdir(parents=True,exist_ok=True)
    episodes,gap_breaks=to_episodes(frames,fps,max_gap)
    counts,probs=transition_matrix(episodes)
    counts.to_csv(OUT/"transition_counts.csv",encoding="utf-8")
    probs.to_csv(OUT/"transition_matrix.csv",encoding="utf-8")
    episodes.to_csv(OUT/"state_episodes.csv",index=False,encoding="utf-8")
    km=km_table(episodes); km.to_csv(OUT/"dwell_survival.csv",index=False,encoding="utf-8")
    ci=sequence_bootstrap(episodes,bootstrap,42); ci.to_csv(OUT/"transition_cluster_bootstrap_ci.csv",index=False,encoding="utf-8")
    # Descriptive sojourn summaries; medians are not claims of causal effects.
    summaries=[]
    for state,g in episodes.groupby("position",sort=True):
        observed=g[g.event_observed & ~g.left_censored].duration_seconds
        summaries.append({"position":state,"episodes":len(g),"observed_exits":int(g.event_observed.sum()),"right_censored":int((~g.event_observed).sum()),"median_observed_dwell_seconds":float(observed.median()) if len(observed) else None,"p90_observed_dwell_seconds":float(observed.quantile(.9)) if len(observed) else None})
    pd.DataFrame(summaries).to_csv(OUT/"dwell_summary.csv",index=False,encoding="utf-8")
    fig,ax=plt.subplots(figsize=(max(7,len(probs.columns)*.65),max(5,len(probs.index)*.5)))
    im=ax.imshow(probs.to_numpy(),vmin=0,vmax=max(.01,float(probs.to_numpy().max())),cmap="viridis",aspect="auto")
    ax.set_xticks(range(len(probs.columns)),probs.columns,rotation=60,ha="right"); ax.set_yticks(range(len(probs.index)),probs.index)
    ax.set_xlabel("Next observed position"); ax.set_ylabel("Current position"); ax.set_title("Observed frame-sequence transitions (row normalized)")
    fig.colorbar(im,ax=ax,label="Empirical transition proportion"); fig.tight_layout(); fig.savefig(OUT/"transition_matrix.png",dpi=160); plt.close(fig)
    meta={**provenance,"fps_used":fps,"max_unobserved_gap_seconds":max_gap,"raw_frame_rows":len(frames),"state_episodes":len(episodes),"observed_transitions":int(episodes.event_observed.sum()),"gap_censored_episodes":int(gap_breaks),"encoded_sequence_prefix_count":int(frames.sequence_id.nunique()),"dataset_page_reported_sparring_sequences":6 if provenance.get("source")=="ViCoS BJJ Positions Dataset" else None,"states":sorted(frames.position.map(normalize_state).unique().tolist()),"bootstrap_cluster_unit":"encoded filename prefix; prefixes may share a physical sparring sequence or camera view","bootstrap_replicates":bootstrap,"interpretation":"Exploratory descriptive estimates from labeled sparring frames; not IBJJF match statistics or causal effects."}
    (OUT/"analysis_summary.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta


def download_annotations(path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".part")
    request=urllib.request.Request(VICOS_URL,headers={"User-Agent":"quantix-bjj-semi-markov/1.0"})
    with urllib.request.urlopen(request,timeout=60) as response, tmp.open("wb") as f:
        total=int(response.headers.get("Content-Length","0")); done=0
        while True:
            chunk=response.read(1024*1024)
            if not chunk: break
            f.write(chunk); done+=len(chunk)
            if done and done%(25*1024*1024)<1024*1024: print(f"Downloaded {done/1024/1024:.0f} MiB / {total/1024/1024:.0f} MiB",flush=True)
    if total and tmp.stat().st_size!=total:
        tmp.unlink(missing_ok=True); raise IOError("Downloaded annotation file size did not match Content-Length")
    tmp.replace(path)


def main() -> None:
    ap=argparse.ArgumentParser(description="Exploratory BJJ transitions and semi-Markov dwell-time summaries")
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo",action="store_true",help="Run only the seeded synthetic demo")
    src.add_argument("--download-vicos",action="store_true",help="Download official ViCoS annotations (CC BY-NC-SA 4.0; non-commercial terms apply)")
    src.add_argument("--annotations",type=Path,help="Analyze an existing ViCoS annotations.json")
    src.add_argument("--event-log",type=Path,help="CSV with sequence_id,frame,position columns")
    ap.add_argument("--fps",type=float,default=FPS)
    ap.add_argument("--max-gap-seconds",type=float,default=2.0)
    ap.add_argument("--bootstrap",type=int,default=500)
    args=ap.parse_args()
    if args.fps<=0 or args.max_gap_seconds<=0 or args.bootstrap<0: ap.error("fps/gap must be positive and bootstrap non-negative")
    if args.demo: frames,meta=make_demo()
    elif args.download_vicos:
        path=RAW/"annotations.json"
        if not path.exists(): download_annotations(path)
        frames,meta=read_vicos(path); meta["source_url"]=VICOS_URL; meta["license"]="CC BY-NC-SA 4.0"
    elif args.annotations:
        frames,meta=read_vicos(args.annotations); meta["source_url"]=VICOS_URL; meta["license"]="CC BY-NC-SA 4.0"
    else:
        frames=pd.read_csv(args.event_log); meta={"source":"User-supplied event/frame log","source_file_name":args.event_log.name,"license":"User must document source and rights"}
    if frames.empty: raise SystemExit("No usable labeled frames found.")
    summary=save_outputs(frames,meta,args.fps,args.max_gap_seconds,args.bootstrap)
    print(json.dumps(summary,indent=2))
    print(f"\nOutputs written to: {OUT}")

if __name__=="__main__": main()
