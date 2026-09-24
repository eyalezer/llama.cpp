# TurboQuant fork comparison

Snapshot: 2026-09-24, using already-fetched remote refs (no fetch performed).

| Branch | Revision | Role |
| --- | --- | --- |
| `turboquant` | `9993986c9` | This fork: upstream through `6e60f3560` (2026-09-23) plus one squashed `Turboquants` commit (author date 2026-08-28; committer date 2026-09-23) |
| `turboquant-origin/feature/turboquant-kv-cache` | `a3d5603d1` | Other fork's default branch, latest commit 2026-09-23 |
| `upstream/master` | `9710a3217` | Fetched upstream, 2026-09-24 |

## How to read this

These are **integration leads**, not a list of patches to cherry-pick in order. The local TurboQuant code is one squash on newer upstream; the other fork diverged from upstream at `15586e2d7` (2026-08-06) and also carries upstream changes. The branches have 851 local-only and 532 other-fork-only commits by ancestry, but those counts do **not** measure missing TurboQuant work. A full tree diff spans 2,023 files, mostly because of the upstream gap. Even a commit that is absent by hash may already be present here via upstream or the local squash.

**Confirmed absent by file check** means the named feature's distinctive files exist on the other fork and not on `turboquant`; it does not prove every related behavior is absent. **Review** means the feature already has some local code, or its effective behavior has not been compared. The first-parent list has 49 entries dated August 28 or later, but this is only a convenient recent-activity window: the local squash was committed on September 23 and already contains Qwen4Exp code, which the other fork merged on August 31. There is no established last-imported commit.

## Largest gaps

| Area | Other-fork references | Local assessment |
| --- | --- | --- |
| Block KV streaming and bounded phase arena | `a3d5603d1` (#357), with `94d91d916` (seq removal), `71e392186` (F16 fallback warning), `cd559a175` (planner cleanup) | **Confirmed absent by file check:** `docs/kv-stream.md`, `src/llama-kv-stream-config.cpp`, `src/llama-kv-stream-plan.cpp`, `tools/kv-stream-bench/`, and KV-stream tests are not in this branch. Review prerequisites, supported backends, and the documented limitations before porting. |
| MoE expert caching, device-routed prefill, CUDA paging | `daa4a32b4` (#327), `cf96a5460` (#322), `bcc337d69` (#364) | **On a separate local branch:** `turboquant-moe-cache` (`25c696523`) has the cache infrastructure, including CUDA, Metal, and Vulkan files. It is absent from `turboquant` / `turboquant-updates`, not missing from local work. Compare the other fork's later CUDA paging rework separately; branch presence alone does not prove it is included. |
| TurboAnchorKV experiment | `80be9a71e` (#333) | **Confirmed absent by file check:** `examples/turboanchorkv/`. An experiment, not a default-runtime requirement. |
| KV and weight backend fixes | `92e0cb507` (#373, Vulkan turbo2/4 KV dequant), `afee2cd88` (SYCL strided KV), `325dbba50` (#372, HIP guard), `c26baf12e` (#342, TQ4_1S decode), `ed0816b03` (#337, RDNA MMQ), `1cd79213b` (#336, CDNA2 MMQ), `9a38e18df` (#318, turbo4 VEC corruption) | **Partial import:** #318's CUDA VEC row selection is in the worktree but cannot be built or run here. Vulkan turbo2/4 dequant shaders were already present. Other backend changes still need review. |
| CUDA fusion, narrow weights, and cache RAM bounds | `1208c5956` (#343), `6d7faf607` (#349), `c80f89732` (#375), `4c41b96c6` (#363), `f193682a2` (#350) | **Partial import:** #343's F32 elementwise chain, #375's scale+bias match ordering, #363's AMD narrow-weight selector, and #350's server cache bound are in the worktree. Shared Q8 activation caching, other fusion follow-ups, and backend-specific speedups remain unported or unverified. |
| Qwen4Exp, DRC, and MTP updates | `5ff15401f` (#324), `b5b813d89` (#340), `4deec5587` (draft-only shared weights), `26d920ee8` (#331), `e3e5d61ae` (#352, GDN replay) | **Review:** `src/models/qwen4exp.cpp` and `conversion/qwen4exp.py` already exist locally. Compare feature details and regression fixes against current upstream model changes. |

## Other changes worth triaging

- CUDA TQ and rotation performance/correctness: `80007e715` (#351, WHT signbits), `27d17bd68` (#348, MMQ recursion), `0fc220298` (#339, Pascal WMMA), `a66580cd5` (#335, MMID coverage), `f73736380` (#334, MMID guard), and `4a54c5207` (no MMQ architecture fallback).
- Flash attention and GPU operation fixes: `a442010c7` (#362), `bef37276b` (HIP D=512 decode), `5ed81c8ae` (HIP large-head guard), and `ee8280b8f` (long-context CUDA temporary allocation). Verify whether upstream's newer FA code already covers each case.
- Tests, portability, and release infrastructure: `68b48398e` (#370, zero-case backend tests), `407f3237b` (#355, MSVC COMDAT), `23fcf04bd` (#353, explicit perplexity `-np`), `186f56e5b` (#329, SYCL Arc), `bc82d72d2` (#382, CUDA 13 CI), `8ca726048` (Windows BoringSSL release), `fb2cc35ab` (CANN CI), and `b59f34027` (default CUDA KV-stream test expectation).
- Documentation and operational warnings: `c80c36fd9` (attention-sink KV sensitivity), `525f7eada` (MoE expert threshold), `461e61a7c` (KV-stream limitations), `1d453f5c4` (KV-stream with MTP). Some only matter if their underlying feature is brought over.

This survey emphasizes the 49 first-parent entries dated August 28 onward, **not** changes proven to have occurred after the local import. Earlier fork-specific work is **not** cleared by this report: for example, ConvRot and Q8_CR work and the initial MoE-cache implementation predate that window. The local squash cannot be matched to an exact other-fork revision by ancestry alone. This survey also does not establish which non-TurboQuant upstream cherry-picks are already semantically present.

## Performance priority

For **CUDA inference with TQ4_1S weights**, review decode kernels first: `c26baf12e` (#342) and, on AMD HIP, `ed0816b03` / `1cd79213b` (#337/#336). Next compare `80007e715` (#351, WHT rotation), then `1208c5956` / `6d7faf607` / `c80f89732` (#343/#349/#375, CUDA fusions and guards). These are plausible token-throughput wins, not measured speedups for this branch; profile the target architecture before porting. For Vulkan, examine `92e0cb507` (#373) only if turbo2/4 KV dequantization is a bottleneck. These CUDA/HIP/Vulkan changes do not imply a Metal speedup.

For **very long single-sequence CUDA contexts that otherwise exceed VRAM**, KV streaming (#357) may matter more than any kernel optimization: the other fork's Qwen3.8-27B memory sweep reached 393K context with an 8 GiB arena while its no-streaming run failed there. It is not a general throughput upgrade: streaming disables prompt-cache save/restore, requires `-np 1`, and the documented default CUDA build can spend much longer in F16 fallback attention. The fork reports 2,914 prefill tokens/s without streaming versus 336 with a 4 GiB arena at ~39.5K prompt tokens on one RTX 5090 with `GGML_CUDA_FA_ALL_QUANTS` off; enabling that build flag changes the tradeoff substantially. See the other fork's `docs/kv-stream.md` and `benchmarks/results/kv-stream-pr-evidence/` for the workload and measurements.

## Recheck and next step

The current uncommitted import needs a Windows CUDA/HIP build before further CUDA cache work is safe. The Metal build verifies `llama-server` and the three new elementwise-chain test graphs, not CUDA dispatch or performance. Run the CUDA `test-backend-ops` cases filtered to `ELEM_CHAIN_FUSION`, compare a representative model's `llama-bench` prefill/decode with `GGML_CUDA_FUSE_CHAIN=0` and its default setting, and check Turbo4 V attention separately on CUDA. Do not infer a speedup from the presence of a fusion path alone.

```sh
git log --first-parent --since=2026-08-28 --date=short --format='%h %ad %s' turboquant-origin/feature/turboquant-kv-cache
git show --stat <other-fork-commit>
git diff turboquant turboquant-origin/feature/turboquant-kv-cache -- <relevant-path>
```

For each area you want, compare its final code and tests with `turboquant` **and** current `upstream/master`, then adapt and test it against the newer base. Do not merge the other fork wholesale: it carries an older upstream base and overlapping code with different history.