# GLM-5.3-Flash on two DGX Sparks

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (zai, 320B MoE, 18B active) as Intel's
[W4A16 AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound) checkpoint, on two DGX Sparks, TP=2 over RDMA.
**43 tok/s average and 64 peak on a thinking-on request, 187 tok/s at 16 streams (210 peak), a 1.18M-token KV pool,
400k context per request.** Three commands.

**Side by side with every other stack:** [myllmbox.com](https://myllmbox.com)

## Measured (2× DGX Spark, RDMA, image v1, the shipped `recipe.yaml`)

| | v1 |
|---|---|
| thinking-on request (pasture), c=1 | **43.1** tok/s average · **63.9** peak |
| peak at c=1 / 2 / 4 / 6 / 8 / 12 / 16 | **56.9 / 79.0 / 105.3 / 128.1 / 153.2 / 177.1 / 210.0** |
| prefill, 128k-token prompt | **1,635** tok/s |
| first token, 1k prompt | **0.92** s |
| KV pool | **1,184,905** tokens (2.96 × a full 400k request) |
| longest prompt served | 388,426 tokens |

**Per prompt** (thinking off, aggregate tok/s of all streams, averages of 3 runs of 300 s with every stream kept busy;
**peak** = the best 10-s window at that concurrency)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 | c=16 |
|---|---|---|---|---|---|---|---|
| **peak** | **56.9** | **79.0** | **105.3** | **128.1** | **153.2** | **177.1** | **210.0** |
| mixed (code + explanation) | 38.5 | 54.5 | 71.0 | 84.7 | 98.1 | 120.4 | 140.6 |
| structured output (JSON schema) | 49.5 | 71.3 | 96.9 | 116.6 | 138.7 | 165.0 | 186.9 |
| long prose (7,000-word story) | 33.7 | 49.6 | 66.7 | 80.0 | 94.5 | 119.7 | 138.4 |
| code (TypeScript) | 39.6 | 56.3 | 73.4 | 88.5 | 102.7 | 126.5 | 147.1 |

### Long context (one request)

Decode = 1,000 tokens after the prompt. Hot = the next turn of the same conversation (prefix cache).

| prompt tokens | first token | hot first token | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1k | **0.92 s** | 0.98 s | 1,315 | 39.4 |
| 8k | 5.0 s | 5.0 s | 1,677 | 39.6 |
| 32k | 20 s | 3.1 s | 1,684 | 39.9 |
| 64k | 39 s | 3.5 s | 1,671 | 39.3 |
| 128k | 80 s | **4.3 s** | **1,635** | 38.5 |
| 192k | 2 min 2 s | 5.1 s | 1,608 | 39.2 |
| 256k | 2 min 47 s | 6.4 s | 1,573 | **38.4** |

## Quality (measured on this serve, thinking on)

lm-evaluation-harness against the running serve, HumanEval complete and a fixed 200-question subset (seed 123123123) of the
others.

| task | questions | score |
|---|---|---|
| HumanEval pass@1 | 164 | **98.8** |
| GSM8K exact match | 200 | **95.5** |
| IFEval prompt-level strict / instruction-level strict | 200 | **91.5** / 94.3 |
| MMLU-Pro | 200 | **83.9** |

## Quick start

```bash
git clone https://github.com/myllmbox/glm53-flash-cluster-recipe.git
cd glm53-flash-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~182G, syncs it, serves on :8000
```

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks with docker +
the NVIDIA container runtime, connected by their ConnectX ports, ssh from the head to the worker (a password once —
`setup.sh` installs a key). With the weights on both boxes a boot serves in about 4 minutes (weights 46 s, all model
loading 79 s); the very first boot also compiles kernels into `./cache`.

**All model configuration lives in [`recipe.yaml`](recipe.yaml)** — image, weights, port, KV budget, every vLLM flag. The
cluster flags and the per-box NCCL/gloo pins come from `cluster.env` (written by `setup.sh`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "zai-org/GLM-5.3-Flash",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Send `"chat_template_kwargs": {"enable_thinking": false}` to turn thinking off per request. Images: up to 8 per request.

## Tuning (recipe.yaml)

- **KV pool.** `kv-cache-memory` is per box: 12G = 1,184,905 pooled tokens. It leaves little room on the head box during
  graph capture; if a boot runs out of memory, set `9000000000`.
- **Context.** `max-model-len: 400000` per request; the pool holds 2.96 such requests at once.
- **Seats.** `max-num-seqs: 16`. The capture list in `compilation-config` reaches 16 × 5 = 80; raise both together.
- **Draft depth.** MTP with up to 4 sampled drafts, the depth chosen live per request (`mtp_depth`, min 1). `mode: off` =
  always 4. `cache/mbx-depth.json` can be edited while the serve runs.
- **Images.** `max-num-batched-tokens: 8192` is also the image budget (a 2k × 2k image is ~5.3k tokens).
- **`host: 127.0.0.1`** keeps the API local; `0.0.0.0` exposes it on the LAN.

## Memory on a Spark

`run.sh` waits for both boxes to have their memory back and evicts its own checkpoint files from the page cache before launch
— no root needed. With root, `./tune-host.sh` sets `vm.compaction_proactiveness=0` on both boxes (worth ~10 % on a serve this
close to the memory edge). `./cache` is per box and deletable; the next boot rebuilds it.

## What's in the image

`myllmbox/glm53-flash-cluster-vllm:v1` — `vllm/vllm-openai:v0.31.0` plus:

1. **our loader** (`load-format: mbx`): 182 GB in about 46 s, the MTP drafter reads only its own 4 GB.
2. **FP8 attention and MLP layers** and a 4-bit drafter head, prepared at load (the INT4 experts stay as published).
3. **dynamic draft depth** in full CUDA graphs.
4. **one-shot RoCE all-reduce** — [b12x](https://github.com/local-inference-lab/b12x)'s RoCE collectives (Apache-2.0) for
   messages up to 2 MB, NCCL above; both halves of each ConnectX-7 port.

Digest: `sha256:440c6bab43145c59d7c4bebb44f5e4379acf38fcbeb49c71b044c39214ea23c2`

## The full box

The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights.

## License

Weights: MIT (zai-org; Intel's AutoRound checkpoint, MIT). Kit scripts: MIT.
