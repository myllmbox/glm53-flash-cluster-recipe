# Optional vLLM patches

Off by default. Turn one on by listing its name in `recipe.yaml`:

```yaml
server:
  patches: my-fix               # patches/my-fix.patch; space-separated, applied in this order
```

At launch `run.sh` copies each file a patch touches out of the image, applies the patch, copies the result to the
worker and mounts it read-only over the image's file on both boxes. The image is never changed; an empty
`patches` = the stock image. A patch that does not fit the image stops `run.sh` before anything starts.

No patches ship with v1 yet.

## Adding one

A unified diff with paths relative to the `vllm` package (`--- a/entrypoints/…`, `+++ b/entrypoints/…`), made against
the image in `recipe.yaml`. Lines before the first `---` are a free-text description.
