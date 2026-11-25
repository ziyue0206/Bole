# The official base appends root-owned toolchain setup to its global profile.
# Keep that setup for root; non-root Bole shells use accessible system tools.
if [ "$(id -u)" = 0 ]; then
    . /opt/bole-upstream-profile
else
    PATH=/opt/bole-tools:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/bin" ]; then
        PATH="$VIRTUAL_ENV/bin:$PATH"
    fi
    export PATH
fi
