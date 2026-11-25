# Third party components

This source tree retains component-specific notices. The components downloaded during preparation retain their upstream terms.

| Component | Source and role | Included material |
| --- | --- | --- |
| RoboDojo | [RoboDojo-Benchmark/RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/f57e1633ffed59761bc8e568cfe7530c4b33381c), revision `f57e1633ffed59761bc8e568cfe7530c4b33381c` | Selected native runtime source under `vendor/robodojo`; MIT text in `licenses/RoboDojo-LICENSE` and the vendored tree |
| XPolicyLab | [XPolicyLab/XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/tree/5d0be1499634d3176258c8f2e31f5fe816497b0a), source submodule revision referenced by the selected RoboDojo commit | Selected client/server code and demo deployment configuration; Apache 2.0 text in `licenses/XPolicyLab-LICENSE` and the vendored tree |
| Moby profiles | [seccomp v0.2.4](https://github.com/moby/profiles/tree/seccomp/v0.2.4) and [AppArmor v0.2.2](https://github.com/moby/profiles/tree/apparmor/v0.2.2) | Derived Codex-specific profiles in `containers/codex`; Apache 2.0 text in `licenses/Moby-profiles-LICENSE` |

The following resources are referenced or downloaded, not bundled in the source archive:

- [OpenAI Codex](https://github.com/openai/codex/tree/rust-v0.169.0) CLI and the matching official bubblewrap helper. Download URLs and archive SHA256 values are recorded in `configs/resources.lock.json`. The helper retains its own upstream component terms.
- [OpenAI codex-universal](https://github.com/openai/codex-universal), used as the official base for the thin Codex runtime layer.
- [RoboDojo assets](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) at the manifest's fixed revision. Asset/data terms are separate from RoboDojo source-code terms; the dataset card identifies Apache 2.0. Preserve any additional component notices supplied upstream.
- NVIDIA CUDA, Isaac Sim, IsaacLab, cuRobo and other software installed by the upstream RoboDojo Dockerfile. Refer to the selected repositories and [NVIDIA license documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses.html).
- Python, PyYAML, NumPy, Pillow, FFmpeg and other dependencies installed in host or container environments. Their distribution metadata and upstream notices remain applicable.

The selected source revisions are provenance records, not a claim of official benchmark endorsement. No simulator assets, model checkpoints, executable release archives, or container layers are shipped in this source directory.
