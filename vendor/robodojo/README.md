# Vendored RoboDojo runtime subset

This directory contains the selected native source Bole needs for task planning and runtime snapshots. Its source reference is RoboDojo revision `f57e1633ffed59761bc8e568cfe7530c4b33381c` from the [official repository](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/f57e1633ffed59761bc8e568cfe7530c4b33381c).

It includes native task/environment utilities, the required XPolicyLab client subset, and their retained license files. It does not include assets, model weights, third-party simulation installations, or the complete upstream Docker build context.

For a new simulation image, use Bole's `source` and `simulator-image` preparation steps. They obtain the full selected upstream checkout and its pinned submodules separately under the configured data directory. Do not build the simulator from this vendored subset.

See [Bole setup](../../docs/getting-started.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md). RoboDojo source retains the license in [`LICENSE`](LICENSE); the XPolicyLab subset retains [`XPolicyLab/LICENSE`](XPolicyLab/LICENSE).
