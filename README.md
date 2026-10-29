# GLM-5.3-Flash on two DGX Sparks

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (zai, 320B MoE, 18B active) as NVIDIA's
[FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) checkpoint (NVFP4 experts, MXFP8 dense layers, weights as
shipped), on two DGX Sparks, TP=2 over RDMA. **45 tok/s average and 87 peak on a thinking-on request, 214 tok/s peak at
16 streams, a 1.15M-token KV pool, 400k context per request.** Three commands.

**Side by side with every other stack:** [myllmbox.com](https://myllmbox.com)

## Measured (2× DGX Spark, RDMA, image v3, the shipped `recipe.yaml`)

| | v3 | v2 |
|---|---|---|
| thinking-on request (pasture), c=1 | **44.9** tok/s average · **86.7** peak | 42.0 · 69.3 |
| peak at c=1 / 2 / 4 / 6 / 8 / 12 / 16 | **60.4 / 74.7 / 107.7 / 119.4 / 144.6 / 173.9 / 214.2** | 64.4 / 65.2 / 95.8 / 111.2 / 126.1 / 157.3 / — |
| prefill, 128k-token prompt | **1,836** tok/s | 1,459 |
| first token, 1k prompt | **1.13** s · **0.26** s cached | 1.02 s |
| KV pool | **1,145,238** tokens (2.86 × a full 400k request) | 1,074,782 |

**Per prompt** (aggregate tok/s of all streams, averages of 3 runs of 300 s with every stream kept busy;
**peak** = the best 10-s window at that concurrency)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 | c=16 |
|---|---|---|---|---|---|---|---|
| **peak** | **60.4** | **74.7** | **107.7** | **119.4** | **144.6** | **173.9** | **214.2** |
| mixed (code + explanation) | 40.7 | 52.7 | 70.5 | 83.0 | 94.5 | 118.5 | 138.5 |
| structured output (JSON schema) | 45.6 | 64.6 | 85.8 | 102.1 | 116.1 | 145.9 | 169.7 |
| long prose (7,000-word story) | 34.4 | 49.8 | 68.3 | 82.8 | 94.4 | 117.9 | 139.1 |
| code (TypeScript) | 41.1 | 54.6 | 71.9 | 84.4 | 100.3 | 125.1 | 146.2 |

### Long context (one request)

Decode = 1,000 tokens after the prompt. Hot = the next turn of the same conversation (prefix cache).

| prompt tokens | first token | hot first token | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1k | **1.13 s** | 0.26 s | 1,082 | 39.9 |
| 8k | 4.6 s | 0.27 s | 1,842 | 37.8 |
| 32k | 17.5 s | 0.36 s | 1,879 | 39.2 |
| 64k | 35.4 s | 0.37 s | 1,859 | 40.7 |
| 128k | 71.5 s | 0.54 s | 1,836 | 41.6 |
| 256k | 148.8 s | 0.80 s | 1,763 | 41.8 |

## Tuning (recipe.yaml)

- **KV pool.** `kv-cache-memory` is per box: 12G = 1,145,238 pooled tokens. It leaves little room on the head box during
  graph capture; if a boot runs out of memory, set `11000000000`.
- **Context.** `max-model-len: 400000` per request; the pool holds 2.86 such requests at once.
- **Seats.** `max-num-seqs: 16`. The capture list in `compilation-config` covers exactly the shapes the depth ceiling allows
  up to 16 requests; raise both together.
- **Draft depth.** MTP with up to 9 sampled drafts, the depth chosen live per request (`MBX_DEPTH_*` in `env`). The ceiling
  `MBX_DEPTH_CEIL` caps the deepest draft by live requests: 1 → 9, 2–3 → 7, 4–7 → 6, 8+ → 5. `MBX_DEPTH_MODE: off` = always 9.
- **Images.** `max-num-batched-tokens: 8192` is also the image budget (a 2k × 2k image is ~5.3k tokens).
- **`host: 0.0.0.0`** serves the API on the LAN (`<box IP>:<port>`); `127.0.0.1` keeps it on the head box only.
- **Patches.** `patches` (server): optional vLLM patches from [`patches/`](patches/), off by default — `patches: hermes-chat`
  for the Hermes agent (after [@yume-arasaki](https://github.com/yume-arasaki)'s patch for the Qwen kit). Applied at launch
  over the image's files; the image itself is unchanged.

## Memory on a Spark

`run.sh` waits for both boxes to have their memory back and evicts its own checkpoint files from the page cache before launch
— no root needed. With root, `./tune-host.sh` sets `vm.compaction_proactiveness=0` on both boxes (worth ~10 % on a serve this
close to the memory edge). `./cache` is per box and deletable; the next boot rebuilds it.

## What's in the image

`myllmbox/glm53-flash-cluster-vllm:v3` — `vllm/vllm-openai:v0.32.0` plus:

1. **our loader** (`load-format: mbx`): 177 GB in about a minute; the MTP drafter reads only its own shards.
2. **NVFP4 experts on Marlin with compressed scales** kept on the GPU (~4 GB per box back for the KV pool); MXFP8 dense layers
   on b12x, as NVIDIA ships them. The weights stay as NVIDIA ships them.
3. **dynamic draft depth 1–9** in full CUDA graphs, capped by the number of live requests; a two-stage draft head.
4. **one-shot RoCE all-reduce** — [b12x](https://github.com/local-inference-lab/b12x)'s RoCE collectives (Apache-2.0) for
   messages up to 2 MB, NCCL above; both halves of each ConnectX-7 port.

Digest: `sha256:313396bd016711f70f2c5daed754a2ef4bae3b61d3199a145e0217404a8c54e2`

v2 (MTP 1–6, MXFP8 on Marlin) stays available: tag `v2` of this repo, image `myllmbox/glm53-flash-cluster-vllm:v2`.
v1 (Intel's W4A16 AutoRound on vLLM 0.31) stays available: tag `v1` of this repo, image `myllmbox/glm53-flash-cluster-vllm:v1`.

## The full box

The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights.

## License

Weights: zai-org GLM-5.3-Flash (MIT); NVIDIA's FP4-Local checkpoint under the NVIDIA Open Model Agreement. Kit scripts: MIT.
