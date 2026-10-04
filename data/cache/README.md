Cached model responses, one JSON file per request, keyed by model, prompt version, prompt
hash, input digest and sample index. Written by live runs (`hlzf run --live`) and committed, so
`HLZF_OFFLINE=1` replays a run without an API key. Each file stores the response text, token
usage, cost and latency of the original call, never the prompt or the API key.
