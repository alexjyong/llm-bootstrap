# DFlash / DSpark Speculative Decoding — Research Notes

Status: **DFlash is implemented** as `--dflash` in `setup_llamacpp.sh` and `docker/setup_docker.sh` (plus wizard support in `llm.sh`), using the self-conversion approach documented below — no third-party GGUF is downloaded. DSpark (open PR, builds directly on DFlash) remains research-only/future work; see the dedicated section further down.

Two Unsloth staff members made forward-looking Discord comments (2026-07-08) that read as general Docker infrastructure work, not a confirmed DFlash release — see "Unsloth staff are working on *something*" below for why that's a weaker signal than it first looked. Worth a quick check of `huggingface.co/unsloth` for a native DFlash release before assuming this doc's workaround is still the best option, but nothing here should be read as "an official release is imminent."

## What is DFlash

DFlash is a speculative-decoding method added to llama.cpp in [ggml-org/llama.cpp#22105](https://github.com/ggml-org/llama.cpp/pull/22105) (merged into `master` 2026-06-28). Like MTP (already supported here via `--mtp`), it pairs a small draft model with the full target model to skip ahead multiple tokens per forward pass. The difference from MTP:

- **MTP**: draft heads are baked into the same GGUF as the target model. No second file, no `-md` flag — just `--spec-type draft-mtp --spec-draft-n-max 3`.
- **DFlash**: draft is a **separate model/GGUF file**, loaded via `-md <path>` alongside `--spec-type draft-dflash --spec-draft-n-max N`. The draft model produces a whole block of candidate tokens in one forward pass (vs. EAGLE3-style one-token-at-a-time drafting), which is what gives it a higher ceiling (author claims up to 8x on Qwen3; the third-party [DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) benchmark reports 3.75x on Qwen3.6-27B with matching pass@1 on MATH-500). At temperature 0 it's output-lossless — identical output to running the target alone, just faster.

## Expected win over this repo's current MTP-only setup

This repo's own `--mtp` description claims "~2x faster generation." The lukaLLM guide's controlled benchmark found DFlash alone reaches **~3.75x** on the same target model — ahead of MTP alone on the numbers available. So adding `--dflash` as an alternative (not just a DSpark stepping stone) looks like a real win on its own, independent of anything else in this doc.

### Unverified anecdote: stacking MTP + DFlash together

Community chatter (Unsloth Discord, unattributed/unverifiable, 2026-07-08) from a user "Micah" reports running an **MTP-format GGUF as the target model** (`Qwen3.6-27B-MTP-UD-Q5_K_XL.gguf`) with a **DFlash draft layered on top via `-md`** (`qwen3.6-27b-dflash-IQ4_XS.gguf`), on `ggml-org/llama.cpp:server-cuda13-b9894` (RTX 5090). This isn't something either the merged PR or the lukaLLM guide describes — both treat MTP and DFlash as mutually exclusive alternatives (the lukaLLM compose file runs them as separate services, never together). Reported results:

- Baseline (same quant, no speculative decoding): ~45 tps.
- MTP + DFlash stacked: 60-100 tps typical, spiking toward ~200 tps on repetitive/looping completions (a tiny DFlash drafter predicts repeated patterns almost perfectly — user noted the TPS spike itself could double as a loop-detection signal).
- Cost: had to **reduce context to fit in 32GB VRAM** — MTP's larger target weights plus a second DFlash draft model plus (per the finding above) DFlash's f16-KV-cache requirement stack VRAM pressure three ways at once.
- User also flagged still-broken official Docker builds as the blocker to trying further combos (matches the backend rough edges already noted in PR #22105's post-merge comments) and separately noted no NVFP4+MTP GGUF exists yet from Unsloth (NVFP4 releases are safetensors-only so far).

Plausibility: not implausible — DFlash's target-side feature extraction reads specific target hidden-state layers (the PR references `llama_set_embeddings_layer_inp`/`_nextn`, "nextn" being MTP terminology), so an MTP-native target likely just works as an ordinary DFlash target, with its baked-in MTP heads sitting unused unless `--spec-type` also invokes `draft-mtp`. But this is a single informal report, not a benchmark — treat as a lead worth testing on this repo's hardware, not a validated technique. If pursued, it argues for **not** making `--mtp`/`--dflash` strictly mutually exclusive in the CLI (unlike the initial plan above) — instead, allow `--mtp` to select the MTP-format target GGUF while `--dflash` independently attaches a draft model via `-md`, since the underlying flags don't actually conflict, only `--spec-type` needs to resolve to `draft-dflash` when both are present (the MTP heads riding along unused). This needs real VRAM budgeting before enabling by default on L4/24GB, given the reported need to already shrink context on a 32GB card.

## Compatibility requirements (from the PR and the guide repo)

- Draft GGUF must report `gguf.architecture == dflash` (older `dflash-draft` GGUFs fail to load — "unknown model architecture"). Check via `https://huggingface.co/api/models/<repo>` before downloading anything.
- Draft and target must share the same base vocab — llama.cpp checks this at load time, but a mismatched draft (e.g., a Gemma-based draft with a 32K vocab against a ~300K-vocab target) is a real failure mode called out in PR comments. Draft *quantization* does not need to match the target's; only the vocab does.
- Hybrid/SSM target models (Qwen3.5-family) can't drop rejected suffixes cleanly (`seq_rm` doesn't decompose recurrent state) — this repo's Qwen3.5-122B-A10B option is a poor DFlash candidate until the follow-up PRs (#19493, #22227) land. Qwen3.6-27B is pure-attention and the intended target.
- MoE targets see smaller speedups (more experts activate during batched verification than during single-token decode) — another reason to scope this to the dense 27B model first, not the 35B-A3B MoE.

## Verified draft GGUF repos for Qwen 3.6-27B

Checked every `Qwen3.6-27B-DFlash*-GGUF` repo on Hugging Face directly (HF API + raw GGUF header parse of `general.architecture`), since HF's download/like counts turned out to be actively misleading here. PR #22105 renamed the arch tag from `dflash-draft` to `dflash` on merge (2026-06-28); repos converted before that date still report the old tag and **fail to load** ("unknown model architecture") on current llama.cpp, but nothing on the HF listing page tells you that.

**Usable (`general.architecture == dflash`):**

| Repo | License | Quants |
|---|---|---|
| `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` | apache-2.0 | BF16, Q4_K_M, Q5_K, Q6_K, Q8_0 |
| `williamliao/qwen3.6-27B-DFlash-GGUF` | apache-2.0 | F16, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0 (widest selection) |
| `jojohai/Qwen3.6-27B-DFlash-GGUF` | mit | IQ4_XS only |
| `giocom/Qwen3.6-27B-DFlash-GGUF` | unset — check before use | BF16, Q4_K_M, Q8_0 |
| `mayreh/Qwen3.6-27B-DFlash-GGUF` | unset — check before use | Q4_K_M only (odd internal filename, low downloads — lowest confidence) |

**Stale — reports the pre-rename architecture, will error on load despite looking popular:**

- `Anbeeld/Qwen3.6-27B-DFlash-GGUF` — most downloads (11k+) and 24 likes of any repo in this search, but broken (`dflash-draft`).
- `spiritbuun/Qwen3.6-27B-DFlash-GGUF` — highest-liked (70), also broken.
- `Lucebox/Qwen3.6-27B-DFlash-GGUF` — reports `qwen35-dflash-draft`, mismatched on two axes.
- `Radamanthys11/Qwen3.6-27B-DFlash-GGUF`, `Ardenzard/Qwen3.6-27B-DFlash-GGUF` — same stale-arch issue.

**Takeaway**: default to `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` or `williamliao/qwen3.6-27B-DFlash-GGUF` (both apache-2.0, multiple quants, verified correct arch) rather than whatever sorts to the top by popularity. This reinforces the compatibility-check requirement above — an automated setup script must check `general.architecture` itself before download, not trust a repo name or quant tag.

### Independent confirmation (Unsloth Discord, 2026-07-08)

A real user hit exactly the failure mode predicted above, independent of this research: "Micah" reported the Lucebox GGUF "didn't work (errors on load)" and got `unknown model architecture: 'dflash-draft'` on another download, even after updating to the latest llama.cpp — matching the stale pre-rename repos flagged in the table above. A community member's advice at that point: "I would try with GGUFs made after that PR was merged, or better yet, download the safetensors and quantize it yourself with the instructions from the PR" — independent validation of the self-conversion approach described below, arrived at by someone else hitting the same wall.

On the positive side, Micah's final working setup used `qwen3.6-27b-dflash-IQ4_XS.gguf` — the only verified-working repo above that ships an IQ4_XS quant is `jojohai/Qwen3.6-27B-DFlash-GGUF`, so that repo now has a real independent user confirming it loads and runs, not just a clean architecture-tag check.

On quant mismatch between draft and target: community consensus in the same thread matches the lukaLLM guide — mismatched quants work, "bigger size = more accuracy, same as any other quant," but going too low on the draft hurts acceptance rate, though one user noted a benchmark ("double linked list test") showed most quants performing "pretty much equal." One interesting unverified hypothesis raised: since a wrong draft guess only costs a discarded speculative pass (never wrong output), a smaller/faster draft with a *lower* acceptance rate could still win on net throughput by getting "more shots on goal" per second — worth testing empirically rather than assuming a bigger draft quant is always better.

### Self-converting the draft model instead of trusting a third party

Since the whole point of the verification table above is not having to trust a random uploader's binary, it's worth noting **self-conversion is not much work**, and is exactly what llama.cpp's own PR documents. Checked the actual converter code (`conversion/qwen.py`, `DFlashModel` class) to confirm what it needs:

1. Download `z-lab/Qwen3.6-27B-DFlash` — the **original, MIT-licensed safetensors release from the actual DFlash authors**, not a re-packager. Small (~2B params, a few GB in bf16).
2. Get *only the tokenizer/config files* from `Qwen/Qwen3.6-27B` for the `--target-model-dir` flag. Confirmed in the converter source: this is read exactly once, inside `set_vocab()`, purely to load the tokenizer (`self.dir_model = self.target_model_dir` around the tokenizer call) — it never touches the target's weight shards. So this step needs only `tokenizer.json`/`tokenizer_config.json`/`config.json` (a few MB), not the full ~54GB of 27B target weights.
3. Run the conversion documented directly in the PR:
   ```
   python convert_hf_to_gguf.py <z-lab-dir> --target-model-dir <qwen-tokenizer-dir> --outtype bf16 --outfile ours.gguf
   ```
   Seconds to a couple minutes on CPU — no GPU, no training.
4. Optionally quantize with stock `llama-quantize` — since `dflash` is now an officially registered architecture (post-merge), no custom tooling is needed, unlike the older pre-merge forks some of these community repos were built with.

The one real added cost: `convert_hf_to_gguf.py` needs llama.cpp's Python conversion deps (`torch` CPU wheel, `transformers`, `sentencepiece`, `protobuf`, `gguf`) — this repo's setup scripts currently avoid any Python ML stack, installing only `huggingface-hub` for downloads. Adding this would be a one-time, moderate dependency install (CPU torch, not CUDA), not a blocker. This could be baked directly into `setup_llamacpp.sh`'s DFlash flow as a self-convert step, sidestepping the third-party-trust question entirely by building from the primary source ourselves.

### Unsloth staff are working on *something*, scope unclear — likely Docker infra, not confirmed to be DFlash

Two Unsloth staff members ("Mike," "Daniel") posted forward-looking comments in the same thread, but re-reading the sequence closely, it's not clear either is specifically about DFlash support rather than general Docker tooling:

- `Mike — 2:06 AM`: "we working on it / should be this week" — a direct reply to an unlabeled image posted by a different user one message earlier, with no text of its own. No way to tell what the image showed; Micah's DFlash troubleshooting happened over an hour earlier (12:40–1:19 AM) and isn't referenced again until 2:43 AM, so this may not even be about the same topic.
- `Daniel — 2:52 AM`: "im working on a new docker process!" — a reply to Micah asking where Docker downloads models to. Reads more like general Docker infrastructure work (download paths, container layout) than a statement about adding DFlash architecture support specifically.

The only place DFlash and "something's being fixed" connect explicitly is Micah's own closing line — "Looking forward to the docker builds to get fixed so I can try it out!" — his hope/inference about what the fix will enable, not a stated commitment from Mike or Daniel that DFlash support is what's landing. Worth a quick check of `huggingface.co/unsloth` for a native DFlash release or updated Docker image before implementing, but don't treat this as confirmed — it's more likely a general Docker Studio/infra fix than dedicated DFlash support.

## Known-working reference configuration (lukaLLM guide)

To directly answer "is there a ready-to-use setup": **yes** — the [lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) repo isn't just a writeup, it's a working Docker Compose setup with a validated model pairing. Pulled its exact `docker/docker-compose.yaml` to extract the concrete, tested config:

- **Image**: `ghcr.io/ggml-org/llama.cpp:server-cuda13` — the official upstream llama.cpp image (CUDA 13), not a self-built one.
- **Target**: unsloth `Qwen3.6-27B-GGUF`, Q4_K_XL quant.
- **Draft**: `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp`, Q8_0 quant — this is the one they actually benchmarked, not just listed as a possibility.
- **Flags**: `-md <draft.gguf> --spec-type draft-dflash --spec-draft-n-max 15 -fa on -np 1 --jinja` (reasoning/thinking disabled, no mmproj), full GPU offload for both target and draft (`-ngl -1` / draft offload "all"), single GPU only — no `--tensor-split` in the compose file.
- **KV cache: f16 for both K and V, explicitly not q8_0** — the compose file comments say q8_0 KV cache caused a **7x slowdown in draft verification speed**. This is a direct conflict with this repo's current default (`q8_0` K/V cache in both `setup_llamacpp.sh` and `docker/setup_docker.sh`) — DFlash would need its own cache-type default, not inherit the existing one.
- **Context**: `LLAMA_CTX=32768` default, well below this repo's 262K default target — chosen specifically because full-size f16 KV cache at large context was "the prime suspect" for earlier server instability.
- Ships a `llamacpp_baseline` service (no speculative decoding) and a `llama_cpp_qwen36_mtp` service side by side for A/B comparison — confirms DFlash and MTP are treated as mutually exclusive alternates in their own setup too (same conclusion as this doc's plan above), never run together, and never sharing a GPU with the baseline.

This means the earlier "open questions" about `--spec-draft-n-max` and KV cache tradeoffs aren't fully open anymore — there's a concrete, tested starting point (`n_max=15` in the compose default, `n_max=12` found best in their separate speed-sweep benchmark; f16 KV cache required, not this repo's default q8_0). What's still untested is whether these hold on this repo's actual hardware (L4/A100, not a 97GB Blackwell card) and with `--tensor-split` multi-GPU.

## Current state of this repo

Two llama.cpp paths already support MTP as a precedent to mirror:

- `setup_llamacpp.sh` — builds llama.cpp from source (shallow clone of `master`, no `git pull` on re-runs — see gotcha below), then runs `llama-server` under systemd.
- `docker/setup_docker.sh` — pulls a pre-built image (`ghcr.io/alexjyong/llm-bootstrap/llama-server:latest`) built by `.github/workflows/docker-build.yml` from `docker/Dockerfile`, which itself does `git clone --depth 1 --branch ${LLAMA_CPP_VERSION}` (defaults to `master`) and compiles.

Both already have `--mtp` wiring (model registry entry, HF repo swap, `EXTRA_FLAGS`/`MTP_FLAGS` for `--spec-type draft-mtp`). DFlash would follow the same shape but needs a second downloaded file instead of a repo swap.

### Gotcha: version pinning

- `setup_llamacpp.sh` only clones once (`if [ -d "$LLAMA_SRC" ]; then echo "Source already cloned"` — no pull). An existing VM's checkout predates the PR merge and won't pick up DFlash without deleting `$WORK_DIR/llama.cpp` or adding an update path.
- The Docker image is built on `workflow_dispatch` only, defaulting to `master` but tagged `:latest` — the image on ghcr.io stays frozen until someone manually re-runs that workflow. DFlash support requires triggering a rebuild after 2026-06-28.

## What would need to change

1. **Rebuild the Docker image** — manually trigger `docker-build.yml` (default `master` already includes DFlash) so `ghcr.io/.../llama-server:latest` has the `--spec-type draft-dflash` code path. Consider pinning `LLAMA_CPP_VERSION` to a specific commit/tag once one is cut, rather than trailing `master` forever.
2. **Add a `git pull` (or re-clone) path to `setup_llamacpp.sh`** so existing installs can pick up DFlash without manual intervention — currently only fresh `$WORK_DIR`s get new llama.cpp code.
3. **Model registry additions** (`setup_llamacpp.sh` and `docker/setup_docker.sh`), parallel to the existing `MODEL_MMPROJ_FILES` array pattern:
   - A `MODEL_DFLASH_HF_REPOS` (or similar) array mapping the Qwen 3.6-27B entry to a validated draft repo. Per the verified-repos table below, default to `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp` or `williamliao/qwen3.6-27B-DFlash-GGUF` — do **not** pick by download/like count, since the most popular repos on HF (`Anbeeld`, `spiritbuun`) are stale pre-rename conversions that fail to load. `z-lab/Qwen3.6-27B-DFlash` is safetensors-only and needs manual `convert_hf_to_gguf.py --target-model-dir <target_hf_dir>` conversion — avoid that path for an automated script; prefer a repo that already publishes GGUF.
   - Since these are individual (non-unsloth) HF uploads, treat this like the existing `--fixed-chat-template` feature: opt-in, documented as third-party/unverified, README should flag it as experimental.
4. **New CLI flag `--dflash`** (mutually exclusive with `--mtp` — both set `--spec-type`, can't combine), following the exact pattern of the existing `--mtp` block:
   - Prompt/flag parsing, restricted to model index 0 (Qwen 3.6-27B) like MTP is today.
   - Download the draft GGUF into `$WORK_DIR/models/` alongside the target.
   - Before downloading, hit `https://huggingface.co/api/models/<repo>` and confirm `gguf.architecture == dflash`, failing fast with a clear error otherwise (this is the #1 reported failure mode in the PR thread).
   - Add `-md $DRAFT_MODEL_PATH --spec-type draft-dflash --spec-draft-n-max 12` to the `ExecStart` block (12 is the guide's best-found value for 27B; make it a `--spec-draft-n-max` override flag rather than hardcoding).
5. **Docker path**: `docker/setup_docker.sh` needs the equivalent flags plus a second bind mount for the draft GGUF (today it only mounts one model file), and the container `command`/`EXTRA_FLAGS` needs `-md`.
6. **KV cache override**: the guide found q8_0 KV cache causes a 7x slowdown in draft verification — DFlash needs to force `--cache-type-k f16 --cache-type-v f16` regardless of the `--kv-cache` flag the user picked, which also means substantially less context fits per GB of VRAM. Combined with the guide's own 32K-context recommendation (full f16 KV cache at large context was "the prime suspect" for earlier server instability), the auto-sized context logic likely needs a lower cap when `--dflash` is set, at least until confirmed stable at this repo's usual 262K target on L4/A100.
7. **Docs**: extend `docs/llamacpp.md` (and `docs/docker.md`) with a DFlash section mirroring the existing MTP section — flags, expected speedup, caveats, and the experimental/third-party-weights disclaimer.

### What actually shipped (deviates from the list above in two ways)

- **Draft sourcing**: self-converts from `z-lab/Qwen3.6-27B-DFlash` + the target's tokenizer via `dflash_convert.sh`, instead of downloading any of the verified third-party GGUF repos. Eliminates the trust question entirely rather than mitigating it. See `dflash_convert.sh`.
- **Context**: no hard cap was added. The lukaLLM guide's 32K limit turned out to be their own troubleshooting hypothesis ("the prime suspect," not a confirmed root cause), not a real llama.cpp/DFlash limitation — baking in an unverified number as a default would have been worse than not capping at all. Instead, `setup_llamacpp.sh`'s existing VRAM-based auto-sizing was corrected to account for forced f16 KV cache and the draft model's VRAM footprint, and lets it compute whatever context actually fits, with an informational (non-blocking) note that very large context is unvalidated.
- Multi-GPU: warns, doesn't block, per the original plan.
- `--mtp`/`--dflash` stacking (see the Discord anecdote below) was **not** implemented — kept mutually exclusive for v1, as originally planned.

## Open questions before implementing

Answered by the lukaLLM reference config above: draft repo/quant to use (`Alittlehammmer`, Q8_0), a starting `--spec-draft-n-max` (15 default / 12 sweep-optimal), and KV cache type (must be f16, q8_0 causes a 7x draft-verification slowdown). Still genuinely open:

- Does DFlash work correctly with this repo's multi-GPU `--tensor-split` path — the guide only tested single-GPU (`-ngl -1`, no tensor split, on a single 97GB Blackwell card). Needs verification on this repo's `l4` (2x L4) preset before enabling by default there; may need to be gated to single-GPU presets (`a100`, `a100-80`) only.
- Whether `n_max=15`/`12` and f16-KV-cache-required still hold on this repo's actual hardware (L4/A100 have far less VRAM headroom than a 97GB Blackwell card — f16 KV cache at even 32K context costs meaningfully more VRAM than this repo's default q8_0/mixed presets budget for).
- Whether to support DFlash + `--thinking` together, and whether the draft model needs its own chat-template handling (unclear from the PR — the draft only predicts tokens, doesn't reformat prompts, so probably fine, but untested here).
- Whether a maintained/official (unsloth or similar) DFlash draft GGUF for Qwen 3.6-27B will ever exist, vs. staying dependent on individual HF uploaders indefinitely.

## DSpark — the newer, unmerged follow-on to DFlash

While researching DFlash, found that DeepSeek published a follow-on technique, **DSpark** ("Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation," DeepSeek + PKU — the paper is now public: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147), submitted 2026-07-06), and it's already been proposed for llama.cpp as [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173) ("spec: add DSpark speculative decoding"). This is **still not merged** as of 2026-07-09, but see the "Update" subsection immediately below — status has changed materially since the paragraph above was first written.

### Update (2026-07-09): PR is much further along, and real numbers landed

- **Active maintainer engagement**, unlike the earlier "no real engagement" status: three llama.cpp maintainers (`am17an`, `ruixiang63`, `ggerganov`) went back and forth with the author (`wjinxu`) over 2026-07-01 through 2026-07-09. The author reworked the implementation twice in response — first removing a standalone `llama_dspark_*` public API (moved the Markov-head computation into the decoder graph instead), then, as of a few hours before this update, **folding DSpark entirely into the existing DFlash code path** rather than keeping it a separate GGUF architecture — a `--spec-type draft-dspark` vs `draft-dflash` flag now just toggles block layout (DSpark reads drafts anchor-first from block position 0; DFlash reads from position 1) on the same underlying implementation. This is a good mergeability signal but it is **still open, not merged**.
- **An independent real-world test exists now** (`AndrewDBN`, 2026-07-08, on Qwen3-8B) and it's a useful reality check on the vendor numbers below: confirmed working, but (a) the draft model file is **not small** — 2.3GB for Qwen3-8B, roughly **half the size of the Q4_K_M target GGUF itself** — contradicting the "small drafter" framing DFlash trades on; (b) measured speedup was **~20% over no speculative decoding** in limited testing, well short of headline multipliers; (c) the author himself disclosed the shipped implementation is still incomplete — confidence head is loaded but unused, confidence-scheduled prefix pruning isn't implemented yet, the Markov chain is greedy-only (breaks under non-greedy sampling), and only Qwen3 backbones are supported (no Gemma4, no DeepSeek-V4 — DeepSeek hasn't open-sourced V4's DSpark weights separately from the model itself).
- **The actual paper's numbers are more modest than the earlier Discord-sourced "1.88x vs 1.55x" figure quoted below** (that appears to have been a single cherry-picked benchmark domain, not the macro-average). Per arXiv:2607.05147's offline benchmarks (Qwen3-4B/8B/14B, macro-averaged over math/code/chat tasks): DSpark beats DFlash on **accepted length** by **+16.3% / +18.4% / +18.3%** respectively, and beats Eagle3 (autoregressive baseline) by +30.9% / +26.7% / +30.0%. Separately, the paper also reports DeepSeek's own **production** deployment of DSpark on DeepSeek-V4 (Flash and Pro previews) vs. their prior MTP-1 baseline under live traffic: 57-85% faster per-user generation at matched throughput — but that's a different (much larger, MoE) model under DeepSeek's own confidence-scheduled, load-aware serving stack, not a number that transfers to a single-GPU llama.cpp deployment of Qwen3.6-27B.
- **Still no official Qwen3.6-27B DSpark draft** from DeepSeek (only Qwen3 4B/8B/14B + Gemma4-12B, per the HF API as of today). Two new community attempts appeared since the "Availability" section below was first written, but neither is usable here: `pablohassan/Qwen3.6-27B-DSpark-FR` (2026-07-07, safetensors-only "speculators"/vLLM format, French-specialized — same target-transferability concern as AEON below, and not a GGUF/llama.cpp path anyway) and `Avesed/Qwen3.6-27B-DSpark` (created 2026-07-09, distilled against `Avesed/Qwen3.6-27B-INT4-W4A16` rather than the stock BF16/GGUF release, safetensors-only, zero downloads/likes — unvalidated by anyone but the uploader).
- **Net effect on the recommendation below: unchanged for now.** Nothing here is actionable yet for this repo, but the trajectory is better than before — the PR looks closer to a mergeable state, and community interest in 27B-scale drafts is picking up even though none fit our stock-model/GGUF/llama.cpp requirements yet. Worth checking again in another week or two.

### How DSpark relates to DFlash

DSpark **reuses the entire DFlash machinery** (encoder/decoder graph, target-layer feature extraction, KV-cache injection, verify/accept path) and adds one thing: instead of DFlash's independent per-position argmax within a drafted block (which causes acceptance to decay across the block since later positions don't condition on earlier ones), DSpark samples the block left-to-right with a small low-rank "Markov" logit bias conditioned on the previously-sampled token. Concretely (original plan, now partially superseded by the "folded into DFlash" rework noted above):

- New GGUF architecture `dspark` (`llama_model_dspark`, a subclass of `llama_model_dflash`) — loads DFlash's weights plus a Markov head (`markov_w1`, `markov_w2`) and an optional (currently unused) confidence head. *(As of 2026-07-09 this standalone arch was merged back into DFlash's implementation — see Update above.)*
- New spec type `--spec-type draft-dspark`, otherwise using the same `-md <draft.gguf>` flag as DFlash.
- Still greedy-lossless — the Markov bias only changes *what's proposed*, not the verify/accept logic.
- Author's own SpeedBench numbers (Qwen3-8B, RTX 4090): DSpark **1.88x** overall decode speedup vs. no draft, vs. DFlash's 1.55x at the same `--spec-draft-n-max 7` — roughly a **1.16-1.21x** win over DFlash across benchmark domains, biggest on reasoning/open-chat tasks. *(See the more conservative macro-averaged paper numbers in the Update above — this PR-description figure looks like a best-case domain, not the average.)*

### Availability for Qwen 3.6-27B specifically

This is the practical blocker right now:

- DeepSeek's own released DSpark drafters are only for `deepseek-ai/dspark_qwen3_{4b,8b,14b}_block7` — **no official Qwen3.6-27B DSpark draft exists**. Still true as of 2026-07-09.
- The only Qwen3.6-27B DSpark-flavored draft found (`Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft`) is fine-tuned specifically against a third-party "AEON" fine-tune/abliteration of Qwen3.6-27B (vocab 248320) and explicitly documented as **not a drop-in drafter for stock `Qwen/Qwen3.6-27B`** — gains "do not transfer to unrelated targets." It's also a vLLM-only artifact (safetensors + vLLM 0.23.0 patch files), not a GGUF/llama.cpp path. Two more recent community repos (`pablohassan/Qwen3.6-27B-DSpark-FR`, `Avesed/Qwen3.6-27B-DSpark`) share the same target-specificity problem — see Update above.
- A working llama.cpp GGUF example does exist for a different model (`ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`), confirming the `-md ... --spec-type draft-dspark` flow works end-to-end today — but only against PR #25173's unmerged branch (its model card literally says "llama.cpp built from `ft-dspark` (or a release that includes `draft-dspark`)").

### What this means for this repo

- DSpark is **not actionable yet** for Qwen 3.6-27B without either (a) training/converting a DSpark draft head ourselves from `deepseek-ai`'s recipe against stock Qwen3.6-27B, or (b) waiting for someone to publish one, plus (c) building from an unmerged PR branch rather than `master` — though that branch is now under active, converging maintainer review rather than sitting untouched, a materially lower risk than when this section was first written.
- Worth revisiting once #25173 merges and/or an official or reputable community DSpark draft for stock Qwen3.6-27B appears. Until then, DFlash (merged, has real drafters for this exact model) is the only realistic near-term win — and per [docs/mtp-vs-dflash-benchmark.md](mtp-vs-dflash-benchmark.md), this repo's own empirical testing found DFlash's real-world edge over `--mtp` is narrower and more context-fragile than the vendor multipliers suggested, so DSpark's incremental gain on top of DFlash (paper: +16-18% accepted length, independent test: ~20% over no-spec-decoding at all) should be read with the same skepticism until tested on our own hardware/model.
- If DFlash support is added per the plan above, the flag/download plumbing (`-md`, registry entry, architecture-verification check) would need only a `--spec-type draft-dspark` swap and a different HF repo to also support DSpark later — no structural rework.

## References

- DFlash upstream PR (merged): https://github.com/ggml-org/llama.cpp/pull/22105
- DFlash benchmark/guide repo: https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP
- DFlash draft weights: https://huggingface.co/z-lab/Qwen3.6-27B-DFlash (safetensors, needs conversion). Verified-working GGUF: `Alittlehammmer/Qwen3.6-27B-DFlash-GGUF-llama.cpp`, `williamliao/qwen3.6-27B-DFlash-GGUF`, `jojohai/Qwen3.6-27B-DFlash-GGUF`. Stale/broken despite popularity — do not use: `Anbeeld/Qwen3.6-27B-DFlash-GGUF`, `spiritbuun/Qwen3.6-27B-DFlash-GGUF`, `Lucebox/Qwen3.6-27B-DFlash-GGUF`
- DSpark upstream PR (open, unmerged, actively reviewed as of 2026-07-09): https://github.com/ggml-org/llama.cpp/pull/25173
- DSpark tracking issue/discussion: https://github.com/ggml-org/llama.cpp/issues/25096, https://github.com/ggml-org/llama.cpp/discussions/25167
- DSpark paper (DeepSeek + PKU, arXiv, 2026-07-06): https://arxiv.org/abs/2607.05147
- DSpark official drafters (Qwen3 4B/8B/14B + Gemma4-12B only, not 27B): `deepseek-ai/dspark_qwen3_{4b,8b,14b}_block7`, `deepseek-ai/dspark_gemma4_12b_block7`
- DeepSeek-V4 production DSpark models (different model family, not directly relevant to this repo): `deepseek-ai/DeepSeek-V4-Flash-DSpark`, `deepseek-ai/DeepSeek-V4-Pro-DSpark`
- DSpark Qwen3.6-27B-AEON draft (not stock-compatible, vLLM-only): https://huggingface.co/Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft
- Newer Qwen3.6-27B DSpark community attempts, neither usable here (see Update above): `pablohassan/Qwen3.6-27B-DSpark-FR` (vLLM/speculators, French-specialized), `Avesed/Qwen3.6-27B-DSpark` (distilled against an INT4-W4A16 variant, unvalidated)
- DSpark llama.cpp GGUF example (different model, unmerged branch): https://huggingface.co/ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF
- This repo's own empirical MTP vs. DFlash benchmark (context for how much to trust vendor multipliers): [docs/mtp-vs-dflash-benchmark.md](mtp-vs-dflash-benchmark.md)
- Existing precedent in this repo: `--mtp` handling in `setup_llamacpp.sh` and `docker/setup_docker.sh`
