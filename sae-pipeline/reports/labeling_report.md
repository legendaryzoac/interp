# S5 — Haiku auto-labeling report

Machine-readable companion: [`labeling_report.json`](./labeling_report.json) (includes the full 384-label map). This is the human-readable audit: the model + cost, the confidence calibration, and spot-checks so label quality is inspectable.

## Summary

| | |
|---|---|
| Route | **Amazon Bedrock** `bedrock-runtime.converse`, ambient AWS creds (IAM user `zachary`) |
| Region | `us-east-1` |
| Model | **Claude Haiku 4.5** — `us.anthropic.claude-haiku-4-5-20251001-v1:0` (US cross-region inference profile, on-demand) |
| Temperature | 0.0 (deterministic-ish, for label consistency) |
| Prompt | per feature: top-10 activating snippets, peak token in «angle brackets», secondary (≥50% max) in ‹angle brackets›, + logit-lens promoted tokens |
| **Features labeled** | **384 / 384** curated (0 null / unparseable) |
| **Confidence** | **176 high / 128 medium / 80 low** |
| API calls | 384 (1 per feature; **0** needed the parse-retry) |
| Token usage | **343,736 in / 10,614 out** (354,350 total) |
| **Estimated cost** | **$0.3968** (Haiku 4.5 @ $1.00/$5.00 per 1M in/out tok) |

Well under the sub-$1 budget. Cost includes the 5-feature `--limit 5` sanity run (cached, never re-paid).

## Method

`label.py` builds one prompt per curated feature from its `top_examples`:

- Each example's token window is rendered as text. The **peak-activating token** (`act_index`) is wrapped in «double angle brackets»; any other token whose decoded activation is ≥50% of that example's max is wrapped in ‹single angle brackets›. Whitespace tokens are escaped (`\n`) so each snippet stays on one line.
- The feature's logit-lens `promoted` tokens are appended as a secondary signal ("tokens this feature pushes the model to predict next").
- Haiku is asked for strict JSON `{"label": <=8 words, "confidence": high|medium|low}`. The parser strips markdown fences and extracts the first `{...}`; on an unparseable reply it retries once, then falls back to `label=null, confidence=low` (logged). **No feature hit that fallback** — all 384 parsed on the first attempt.
- Resumable: every result is cached to `D:/dev/sae-artifacts/L8/labels_cache.json` keyed by feature id, so reruns never re-pay. `--limit N`, `--force`, `--write-only`, `--verify` supported. Concurrency = 4 in flight with exponential backoff on throttling.

## Confidence calibration

**176 high / 128 medium / 80 low.** The split is honest, not inflated: the **low** bucket is dominated by the highest-frequency features (the `high_frequency` selection tier), which are genuinely polysemantic — e.g. features that fire on generic punctuation, sub-word fragments, or function words across unrelated contexts. Haiku correctly declines to force a crisp theme on them. The crisp **high** labels concentrate in the low-/mid-frequency `diversity_*` features (Arabic script, scientific notation, hyphenated compounds, citation brackets) — exactly the pattern S4's harvest report predicted.

## Spot-checks (label vs. top-3 activating snippets)

Format: `[peak activation] snippet`, peak token in «», secondary (≥50%) in ‹›.

**feature 18** — freq 0.0012 (diversity_band4) — label: **"Closing bracket or end of citation/reference"** — confidence: high
  - `[68.4]` , the BAR was issued to soldiers of various heights.[«48»]\n\nAs originally conceived, US Army tactical doctrine called
  - `[68.2]` asaki; causing the Japanese to surrender on September 2, ending World War II.[«173»
  - `[66.7]`  forests and the rolling hills of the Piedmont.[«233»] The Appalachian Mountains divide the eastern seaboard from the

**feature 41** — freq 0.0020 (diversity_band5) — label: **"Forward slash separators in lists and ratios"** — confidence: high
  - `[17.4]` onogatari II Nekketsu-hen BD/«DVD» CM2 by pKjd\n\nSeven-11
  - `[15.3]` <|endoftext|> RB/«LB», UCLA\n\nCoach of the Year: Todd Graham, Arizona State\n\n
  - `[15.1]`  Orchard, Randy Gregory, Markus Golden, Eli Harold as the OLB/«DE»

**feature 87** — freq 0.0001 (diversity_band1) — label: **"Image caption or photo credit markers"** — confidence: high
  - `[82.6]` ; Suspect Has Died\n\nEnlarge this image toggle« caption» DAVID MANNING/Reuters /Landov DAVID MAN
  - `[79.1]` Burning Coverage Conundrum\n\nEnlarge this image toggle« caption» Joe Raedle/Getty Images Joe Raedle/
  - `[79.0]`  Which Might Please Restaurant Workers\n\nEnlarge this image toggle« caption» Courtesy of Packhouse Meats Courtesy of Packhouse Meats

**feature 5710** — freq 0.0002 (diversity_band2) — label: **"Arabic and Persian script characters"** — confidence: high
  - `[38.3]` َ‹ر›‹َ›‹ب›‹ي›‹ْ› - Arabic ��‹ا›«ر»‹س›‹�›� - Persian ����‹ت›‹و› -
  - `[35.7]`  Crescent Idlib�� in Arabic: ��‹ال›‹ه›«ل»‹ال›‹ ال›‹�›�‹�›‹�›‹م›‹ر›‹ا�›�‹ل›‹ب›
  - `[35.6]`  Arabic:‹ م›‹ع›‹�›�‹و›‹ب› ‹ل›‹و›«ن»‹�›�‹ا�›�‎ (January 24, 1956 – June

**feature 14498** — freq 0.0035 (diversity_band5) — label: **"Hyphenated compound adjectives or adjectival suffixes"** — confidence: high
  - `[20.0]` <|endoftext|> on March 8, one of the sunny,« snow»-free days in early spring, and had no trouble finding the
  - `[19.6]` <|endoftext|> support producers still making challenging,« sub»-low beats – was a top seller in Blackmarket Records, one of the
  - `[18.5]`  it seemed. Williams who was noted for her sympathetic,« eyewitness» Letters Written in France had just published a poem in praise of

**feature 19084** — freq 0.0014 (diversity_band4) — label: **"Digits in scientific notation or decimal numbers"** — confidence: high
  - `[33.2]` ‹514›‹54›365e+00, -2.‹317›«43»276e+00, 9.‹416›‹14›308e-
  - `[33.1]` ‹75›‹87›‹16›79e+06 -7.‹68›«40»‹79›51e+06 3.‹05›‹26›‹70›70e
  - `[33.0]` ‹540›‹312›39] [ 1898.‹610›‹79›121] [ 126.‹23›«77»‹68›85] [ -223.‹4›

**feature 11242** — freq 0.0233 (high_frequency) — label: **"Transition words and discourse markers"** — confidence: medium
  - `[6.6]`  specific domain‐related content. (For discussion of this possibility relevant to openness,« see»
  - `[6.1]`  and not to engage in a debate about the specific psychological mechanisms underlying those interactions.« However»‹,› on a
  - `[5.5]`  Study of the Fast-food Industry in New Jersey and Pennsylvania: Reply.�«�» American

**feature 20447** — freq 0.0764 (high_frequency) — label: **"Partial or fragmented words and tokens"** — confidence: **LOW**
  - `[8.8]` Kinnon\n\nLine 2: Johnny G‹aud›reau Sean‹ Mon›ahan Jack‹ E›«ic»
  - `[8.2]` \nEvery year, Salesforce brings in influential speakers — including women like Patricia Ar«qu»
  - `[8.1]` Spiffy_1\n\nOffline\n\n‹Activity›: 234\n\n«Merit»‹:›‹ 100›\n\n‹Full›‹ Member›‹Activity›:

**feature 2917** — freq 0.0440 (high_frequency) — label: **"Punctuation and special characters in structured text"** — confidence: **LOW**
  - `[6.0]`  incident on The Oprah Winfrey Show in 2005, in« which» Cruise repeatedly jumped on the couch next to Oprah, fell to
  - `[4.8]`  Hawthorne Stacy Hawthorne Photo: Vallejo Police Photo: Valle«jo» Police Image 1 of / 1 Caption Close
  - `[4.8]` . This is nothing short of absolutely, positively delicious! I am pretty« certain» that my favorite Potaje de Garban

**feature 19444** — freq 0.0424 (high_frequency) — label: **"Structural markers and formatting delimiters"** — confidence: **LOW**
  - `[6.5]`  did some tests.\n\nTest 1:‹ Pour›‹ 4› cuts‹ of›‹ water›‹ into›‹ the›« bag»‹.›\n
  - `[6.5]` \n\nnamespace test\n\n{\n\nint« min»(int‹ x›, int‹ y›);\n\nclass X {
  - `[5.9]` �s been said about it so far:\n\n��Int‹ense›‹ and›« atmospheric»‹…›INDOCTRIN‹ATION›

**feature 585** — freq 0.0356 (high_frequency) — label: **"Closing punctuation and citation/reference markers"** — confidence: **LOW**
  - `[5.6]` wa Material conditional �� → �� {\displaystyle« \»‹var›‹phi›‹ \›to \psi } C �� �
  - `[5.5]`  Material conditional �� → �� {\displaystyle‹ \›«var»‹phi›‹ \›to \psi } C �� ��
  - `[5.1]`  13 Bryan Coquard (Fra) Team Europ«car» 0:11:26 14 Jay McCarthy (Aus)

**feature 9787** — freq 0.0415 (high_frequency) — label: **"Prepositions and function words in clauses"** — confidence: **LOW**
  - `[7.6]` ��\n\nSánchez was‹ later› asked« by»‹ Sky› Sports‹ about›‹ the› chances of him staying at Arsenal. �
  - `[6.6]` \n‹This› shouldn��t be surprising‹.›‹ After›‹ all›«,» does‹ the› economy feel as if it��s on the
  - `[6.4]`  York Times newsletters.\n\n‹A› number‹ of›‹ the›‹ kids›« have»‹ missed› days‹ at›‹ school›‹ to› show up for court‹ dates›‹ at› which

**Read:** the six high-confidence picks (18, 41, 87, 5710, 14498, 19084) each name the pattern the «marked» token actually shares — citation/bracket ends, `A/B` slash separators, `toggle caption` photo credits, Arabic/Persian glyphs, hyphenated compounds, scientific-notation digits. The five **low** picks (20447, 2917, 19444, 585, 9787) are the high-frequency polysemantic features; their labels stay deliberately generic and are flagged low — the intended behaviour. Feature 11242 (discourse markers) is a reasonable **medium**: a real theme with some noise.

### The 80 low-confidence features

All 80 still carry a non-null label (a best-effort generic description) plus `label_confidence: "low"`, so the Inspector can show them but visibly down-weight them. None are `null` (nothing was unparseable). S8's hand-curated gallery should prefer the high-confidence `diversity_*` features.

## Verification

`label.py --verify` (re-reads the written files):

- Curated features with a `label_confidence` in {high,medium,low}: **384 / 384** (missing: 0).
- Null labels: **0**.
- `index.json` label/confidence == chunk-file label/confidence on a sampled **20** ids: **20/20 agree** (mismatches: []).
- Chunk features carrying a label: **384**.
- Confidence distribution recomputed from `index.json`: {'high': 176, 'medium': 128, 'low': 80}.

## Write-back (what changed on disk)

`label` + `label_confidence` were added to:

1. all 384 per-feature objects in `features_0000.json` (256) + `features_0001.json` (128);
2. the 384 `hasDashboard` entries in `index.json` (the 24,192 non-dashboard entries are untouched).

All files rewritten as compact single-line JSON with `ensure_ascii=True` (matching the S4 emit format), via temp-file + atomic replace.

## Flag for S6 — content_hash re-stamp needed

> The dashboards' content_hash (manifest.json content_hash=36b59552633dccb3, produced in S3 over the encoder/decoder artifacts; the S4 dashboards are not yet hash-stamped) is now STALE for the dashboard JSONs: this story rewrote features_0000.json, features_0001.json and index.json to add label + label_confidence. S6 (publish) must re-stamp the dashboard payload's content_hash when it ships these files. NOT fixed here (model-pipeline's stamper is out of this story's scope).

## Reproduce

```
export HF_HOME=D:/dev/hf-cache PYTHONIOENCODING=utf-8   # AWS creds are ambient
# sanity: label + print the first 5 (no write-back)
D:/dev/sae-venv/Scripts/python.exe label.py --limit 5 --sample 5
# full run: label all 384 (resumes from cache), write back, verify
D:/dev/sae-venv/Scripts/python.exe label.py
# re-merge cache into JSONs only / verify only:
D:/dev/sae-venv/Scripts/python.exe label.py --write-only
D:/dev/sae-venv/Scripts/python.exe label.py --verify
```
