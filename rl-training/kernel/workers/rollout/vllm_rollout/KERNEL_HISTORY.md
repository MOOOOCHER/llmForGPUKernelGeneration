# Opt-in bounded TTS context

Enable on the plain MultiTurnAsyncvLLMEngine / KernelAgent evaluation path:

```text
++actor_rollout_ref.rollout.multi_turn.history_mode=initial+last_kernel
```

The two leading `+` characters are Hydra's add-or-override syntax.
No launcher or prompt defaults are changed. Omitting the setting preserves
existing per-turn history behavior; setting it to `all` forces full history.

Before every generation, the new mode keeps all original messages (including
task, system instructions and injected skills), the latest extracted kernel,
and feedback after that response. It reuses KernelAgent's kernel selection.
Earlier reasoning and implementations are excluded from generation context,
not from saved trajectories. A response without extractable code becomes a
short explicit placeholder rather than reusing an older implementation.

The 32k model limit and existing output-token cap still apply. This mode avoids
history growth across turns, but does not guarantee enough room for arbitrarily
large tasks, kernels or feedback. Run matched fresh evaluations to compare
models; existing results are not modified or relabelled.

Tests: `python3 -m unittest -v test_kernel_history.py` from this directory.
These are CPU-only history/extractor/integration-placement tests, not a GPU
end-to-end generation test. Other engine variants are not modified.
