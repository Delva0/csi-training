# The client hangs on `/v1/embeddings` while the server logs `200 OK`

Verified at vllm-ascend **v0.11.0** (`2f1aed98ccdb`) with vllm **v0.11.0**
(`b8b302cde434`); recorded for **#6424** (bge-m3, 910B, CANN 8.3.RC2, K8s, one
`POST /v1/embeddings` with `encoding_format: "float"`, `/dev/shm` already ruled out by
the reporter). This is class 9 of the topic index: the engine started, the request was
scheduled, nothing raised — and the client still received nothing.

## What the two pasted lines say

1. **The `torch.save` warning is normal and cannot be the cause.** The paste carries
   `(Worker_TP0 pid=411) ('Warning: torch.save with "_use_new_zipfile_serialization =
   False" is not recommended for npu tensor …')`. At this pin that exact text is
   printed inside the *successful* run of the plugin's own embedding tutorial —
   `docs/source/tutorials/single_npu_qwen3_embedding.md:96`, under a
   `(VllmWorker rank=0 pid=…)` prefix, in a block whose output is the correct
   `[[0.747…, 0.075…], [0.088…, 0.631…]]` similarity matrix. It is torch_npu's notice
   about pickling an NPU tensor across a process boundary — the pooling vector leaving
   the worker — and it fires on the success path. One grep on the docs tree closes it;
   state it and never rank it.
2. **The response body is built before the 200 is written, in the API server.**
   `vllm/entrypoints/openai/serving_embedding.py:134` calls
   `EmbeddingRequestOutput.from_base(final_res)` → `EmbeddingOutput.from_base`
   (`vllm/outputs.py:225-230`, `return EmbeddingOutput(pooled_data.tolist())`), and
   `_get_embedding` (`serving_embedding.py:41-54`) returns that list unchanged for
   `encoding_format="float"` (base64 is just `np.array(...).tobytes()`). Whatever
   could go wrong converting the pooled tensor — device sync, dtype, a raise —
   therefore happens *before* the response object exists. A request the access log
   reports as `200 OK` reached the end of this path.
3. **"200 OK, no client bytes" is a transport statement.** uvicorn writes its access
   line only after the application completed the response, and prints
   `ASGI callable returned without completing response.` when it did not (recalled
   from uvicorn — uvicorn is not in this snapshot, so treat it as the working
   assumption and let check 1 below settle it). The `10.16.x.x:0` peer port in the
   reporter's own line is what that server logs for a peer that arrived through
   `X-Forwarded-For` (the proxy-headers middleware leaves the port at 0) — i.e. a hop
   existed between their curl and this access line.

## What the release does and does not validate

bge-m3 is a supported pooling checkpoint at this pin: `tests/e2e/singlecard/test_bge_model.py`
serves it with `task="embed"`, `enforce_eager=True` and compares against HF by cosine,
and the v0.11.0rc1 checklist ticks "pooling model: bge-m3, bge-reranker" (#3927). Both
are the **offline** `LLM.encode` path. Nothing in the tree drives bge-m3 through HTTP
`/v1/embeddings`, so "supported" must not be read as "this endpoint was exercised" —
say both, and do not conclude the model is unsupported (the earlier bge-m3 reports are
#235 new-model and #255 "the same vector for every input", neither a transport
symptom).

## Ranked causes, each with the one check

1. **The hop between the pod and the client** — K8s Service / Ingress / proxy /
   service-mesh sidecar: response buffering, `proxy_read_timeout`, keep-alive or a
   half-open connection. Check: the identical `curl` **inside the pod** against
   `http://127.0.0.1:8000/v1/embeddings`, then `kubectl port-forward` from the
   workstation, then through the Service and the Ingress. The first layer that
   reproduces the hang owns it, and the in-pod run is also the control that exonerates
   vllm-ascend.
2. **Path MTU/MSS on the overlay.** bge-m3's dense head is 1024 floats ≈ 20 KB of JSON
   while `/v1/models` is ~1 KB; small-works/large-hangs is the black-holed-PMTU
   signature. Check: both requests over the same path, plus `curl -v` on the client —
   a response head with a truncated body is transport; no head at all means the
   request never came back.
3. **The request really is still in flight and the 200 belongs to another
   connection** (a proxy probe, an earlier retry). This is the only lane that puts the
   fault back on the Ascend side, and check 1 decides it: loopback hangs too → the
   pooling lanes reopen (index classes 2 and 7, `batch-index-desync.md`,
   `prefix-cache-hit-embedding-drift.md`); loopback returns the vector → they are out.

## Do not rebuild the serialization story

Where the tensor is pickled (`vllm/v1/serial_utils.py`'s custom-type map), whether the
frontend unpickles it back onto a device, and what `torch.save`'s legacy path does with
an NPU storage are all *upstream* of the response body and cannot produce a logged 200
with no bytes — fact 2 is the reason. Recorded #6424 spent 8 assistant turns and 12
tool calls (~55 K characters of reasoning, one 363-character message, cut at the
16 384-token cap) inside exactly that question, after it had already found the
tutorial line proving the warning benign. The deliverable was the classification
sentence, the red-herring closure and one loopback curl.

## Delivery

First sentence: the server completed the response, so nothing on the NPU path failed.
Then the warning closure with the tutorial `path:line`, the ranked list, and two
artefacts to send back — `curl -sv` from the client and the same curl from inside the
pod — plus `data[0].embedding`'s length and `usage.prompt_tokens` from the first
response that does arrive (1024 floats for bge-m3's dense head), so the first working
answer is comparable with the reference service.
