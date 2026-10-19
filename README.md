# GLM-5.3-Flash on two DGX Sparks

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (zai, 320B MoE, 18B active) as NVIDIA's
[FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) checkpoint (NVFP4 experts, MXFP8 dense layers, weights as
shipped), on two DGX Sparks, TP=2 over RDMA. **42 tok/s average and 69 peak on a thinking-on request, 157 tok/s peak at
12 streams, a 1.07M-token KV pool, 400k context per request.** Three commands.

**Side by side with every other stack:** [myllmbox.com](https://myllmbox.com)

## Measured (2× DGX Spark, RDMA, image v2, the shipped `recipe.yaml`)

| | v2 |
|---|---|
| thinking-on request (pasture), c=1 | **42.0** tok/s average · **69.3** peak |
| peak at c=1 / 2 / 4 / 6 / 8 / 12 | **64.4 / 65.2 / 95.8 / 111.2 / 126.1 / 157.3** |
| prefill, 128k-token prompt | **1,459** tok/s |
| first token, 1k prompt | **1.02** s |
| KV pool | **1,074,782** tokens (2.69 × a full 400k request) |

**Per prompt** (aggregate tok/s of all streams, averages of 3 runs of 300 s with every stream kept busy;
**peak** = the best 10-s window at that concurrency)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| **peak** | **64.4** | **65.2** | **95.8** | **111.2** | **126.1** | **157.3** |
| mixed (code + explanation) | 37.8 | 51.2 | 67.0 | 80.9 | 93.4 | 114.4 |
| structured output (JSON schema) | 43.0 | 58.4 | 82.2 | 98.7 | 110.4 | 134.2 |
| long prose (7,000-word story) | 33.8 | 47.3 | 65.9 | 79.1 | 91.2 | 113.2 |
| code (TypeScript) | 40.3 | 49.0 | 64.9 | 81.7 | 94.7 | 118.6 |

### Long context (one request)

Decode = 1,000 tokens after the prompt. Hot = the next turn of the same conversation (prefix cache).

| prompt tokens | first token | hot first token | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1k | **1.02 s** | 1.02 s | 1,190 | 37.4 |
| 8k | 5.6 s | 5.7 s | 1,485 | 38.3 |
| 32k | 22 s | 3.5 s | 1,494 | 38.2 |
| 64k | 44 s | 3.9 s | 1,480 | 39.5 |
| 128k | 1 min 30 s | **4.8 s** | **1,459** | 40.9 |
| 192k | 2 min 17 s | 5.6 s | 1,436 | 36.5 |
| 256k | 3 min 5 s | 7.0 s | 1,416 | **35.7** |

## Quality (NVIDIA's model card)

| precision | GPQA Diamond | ToolEvalBench |
|---|---|---|
| BF16 | 92.1 | 91.9 |
| **FP4-Local** (this kit's weights) | **91.3** | **92.4** |

## Quick start

```bash
git clone https://github.com/myllmbox/glm53-flash-cluster-recipe.git
cd glm53-flash-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~166G, syncs it, serves on :8000
```

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks with docker +
the NVIDIA container runtime, connected by their ConnectX ports, ssh from the head to the worker (a password once —
`setup.sh` installs a key). With the weights on both boxes a boot serves in about 9 minutes, the first one included (the
weights load in about 4.5 min; kernels compile into `./cache`).

**All model configuration lives in [`recipe.yaml`](recipe.yaml)** — image, weights, port, KV budget, every vLLM flag. The
cluster flags and the per-box NCCL/gloo pins come from `cluster.env` (written by `setup.sh`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "zai-org/GLM-5.3-Flash",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

GLM-5.3-Flash always thinks. Set the depth per request with `"reasoning_effort": "low"` or `"high"` (default: max).
Do not send `enable_thinking: false`: the model still thinks and the reasoning lands in the answer. Images: up to 8 per
request.

## Tuning (recipe.yaml)

- **KV pool.** `kv-cache-memory` is per box: 12G = 1,074,782 pooled tokens. It leaves little room on the head box during
  graph capture; if a boot runs out of memory, set `9000000000`.
- **Context.** `max-model-len: 400000` per request; the pool holds 2.69 such requests at once.
- **Seats.** `max-num-seqs: 16`. The capture list in `compilation-config` reaches 16 × 7 = 112; raise both together.
- **Draft depth.** MTP with up to 6 sampled drafts, the depth chosen live per request (`mtp_depth`, min 1). `mode: off` =
  always 6. `cache/mbx-depth.json` can be edited while the serve runs.
- **Images.** `max-num-batched-tokens: 8192` is also the image budget (a 2k × 2k image is ~5.3k tokens).
- **`host: 127.0.0.1`** keeps the API local; `0.0.0.0` exposes it on the LAN.
- **Patches.** `patches` (server): optional vLLM patches from [`patches/`](patches/), off by default — `patches: hermes-chat`
  for the Hermes agent (after [@yume-arasaki](https://github.com/yume-arasaki)'s patch for the Qwen kit). Applied at launch
  over the image's files; the image itself is unchanged.

## Memory on a Spark

`run.sh` waits for both boxes to have their memory back and evicts its own checkpoint files from the page cache before launch
— no root needed. With root, `./tune-host.sh` sets `vm.compaction_proactiveness=0` on both boxes (worth ~10 % on a serve this
close to the memory edge). `./cache` is per box and deletable; the next boot rebuilds it.

## What's in the image

`myllmbox/glm53-flash-cluster-vllm:v2` — `vllm/vllm-openai:v0.32.0` plus:

1. **our loader** (`load-format: mbx`): the MTP drafter reads only its own shards.
2. **NVFP4 experts on Marlin with compressed scales** kept on the GPU (~4 GB per box back for the KV pool); MXFP8 layers on
   Marlin. The weights stay as NVIDIA ships them.
3. **dynamic draft depth 1–6** in full CUDA graphs.
4. **one-shot RoCE all-reduce** — [b12x](https://github.com/local-inference-lab/b12x)'s RoCE collectives (Apache-2.0) for
   messages up to 2 MB, NCCL above; both halves of each ConnectX-7 port.

Digest: `sha256:9e06b17d349d4723a01bec0c3d163f55bdc50683b0edb6a0c6c99f9ea9901c95`

v1 (Intel's W4A16 AutoRound on vLLM 0.31) stays available: tag `v1` of this repo, image `myllmbox/glm53-flash-cluster-vllm:v1`.

## The full box

The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights.

## License

Weights: zai-org GLM-5.3-Flash (MIT); NVIDIA's FP4-Local checkpoint under the NVIDIA Open Model Agreement. Kit scripts: MIT.
