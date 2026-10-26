# Quantix BJJ Semi-Markov Analysis

An exploratory, reproducible pipeline for converting timestamped Brazilian jiu-jitsu position labels into state episodes, empirical transition matrices, dwell-time summaries, and Kaplan-Meier-style sojourn curves.

## What the data supports

The included real-data adapter targets the **University of Ljubljana ViCoS Brazilian Jiu-Jitsu Positions Dataset**. Its official page describes 120,279 labeled frames from **six sparring sequences**, position labels across 18 classes, and frame metadata. The annotation filenames yield 16 two-digit prefixes in this download; the mapping between those prefixes, physical sparring sequences, and camera views is not documented by the dataset page. They must not be treated as 16 independent matches. This is frame-level sparring data, **not IBJJF tournament event logs**. The outputs describe observed label changes in these sequences; they do not estimate competitive win rates, submission success, causal effects, or population-wide BJJ strategy.

The annotations carry a **CC BY-NC-SA 4.0** license. The data is downloaded locally and is excluded from Git. Attribute the dataset and paper, preserve the same license for adapted annotations/results, and check the license before any commercial use. Dataset details and citation: <https://vicos.si/resources/jiujitsu/>.

## Quick start

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Smoke/demo run. This uses seeded synthetic sequences, never real athletes.
python main.py --demo

# Download the official annotations to data/raw/ and analyze them.
python main.py --download-vicos
```

The annotation file is about 200 MiB. For an already downloaded copy:

```powershell
python main.py --annotations data/raw/annotations.json
```

For another lawfully obtained source, provide a CSV with `sequence_id`, `frame`, and `position` columns:

```powershell
python main.py --event-log path\to\event_log.csv --fps 30 --max-gap-seconds 2
```

## Outputs

- `transition_counts.csv` and `transition_matrix.csv`: observed adjacent-label transitions and row-normalized proportions.
- `state_episodes.csv`: run-length encoded position episodes, their observed duration, next state, and censoring flags.
- `dwell_summary.csv` and `dwell_survival.csv`: descriptive sojourn summaries and a non-parametric survival table. The first episode of each source sequence is excluded from the survival table as left-censored; final and gap-interrupted episodes are right-censored.
- `transition_cluster_bootstrap_ci.csv`: percentile intervals resampling whole source sequences. With only six sequences, these intervals are exploratory and potentially unstable.
- `transition_matrix.png` and `analysis_summary.json`.

The default 30 FPS and 2-second gap threshold are configurable analysis choices. Inspect frame-gap distributions and label definitions before interpreting dwell times. The code treats the two-digit image prefix as a file-sequence cluster and the annotation's `frame` field as the time index. The trailing `1`/`2` in labels such as `mount2` is retained because the source uses it to identify which athlete is in top position.

## Methods and limitations

A row-normalized transition matrix is a descriptive first-order Markov summary. The semi-Markov extension retains observed state dwell times rather than assuming geometric waiting times. The Kaplan-Meier-style table handles right-censoring at sequence ends and long observation gaps; it excludes left-censored starting episodes. It is not a Cox model and makes no causal claim.

The source describes six sparring sequences; adjacent video frames are highly dependent, and filename prefixes may correspond to camera views or segments rather than independent bouts. State labels were manually annotated and the source authors caution that pose/label correctness is not guaranteed. Cluster bootstrap intervals over filename prefixes are exploratory and may understate uncertainty if prefixes share a bout. Do not treat 120k frames as 120k independent bouts, and do not call these IBJJF results.

BJJGraph is a separate authored instructional state graph with estimated probabilities, not observed competition outcomes; this project does not use its weights as empirical data. Its source repository describes its data as noncommercial. See <https://github.com/diogoseca/bjjgraph> and honor its license if you use it.

## Data citation

Hudovernik, V. and Skočaj, D. (2022). *Video-Based Detection of Combat Positions and Automatic Scoring in Jiu-jitsu*. Proceedings of the ACM Multimedia Workshop MMSports '22. Cite the dataset authors and follow the official dataset page's license terms.

## Code license

Project code is MIT licensed. Dataset files and adaptations are governed separately by their source license; no dataset files are committed here.
