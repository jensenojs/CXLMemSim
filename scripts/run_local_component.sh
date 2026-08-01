#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf '%s\n' "usage: $0 --profile PROFILE --execution-root DIR --cache-root DIR" >&2
}

profile=
execution_root=
cache_root=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) [[ $# -ge 2 ]] || die "--profile requires a path"; profile=$2; shift 2 ;;
        --execution-root) [[ $# -ge 2 ]] || die "--execution-root requires a path"; execution_root=$2; shift 2 ;;
        --cache-root) [[ $# -ge 2 ]] || die "--cache-root requires a path"; cache_root=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ -n $profile && -n $execution_root && -n $cache_root ]] || { usage; die "all arguments are required"; }

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $profile == "$ROOT/manifests/build-profile.json" ]] || die "profile does not match the component contract"
[[ ! -e $execution_root ]] || die "execution root already exists: $execution_root"
[[ -d $cache_root ]] || die "cache root is not a directory: $cache_root"
readonly OUTPUTS=$execution_root/outputs
mkdir -p "$OUTPUTS"
bash "$ROOT/scripts/build_component.sh" \
    --work-dir "$execution_root/build" --cache-dir "$cache_root" \
    --payload-dir "$OUTPUTS/payload"
bash "$ROOT/scripts/package_component.sh" \
    --payload-dir "$OUTPUTS/payload" --source-commit "$(git -C "$ROOT" rev-parse HEAD)" \
    --work-dir "$execution_root/package" --manifest-out "$OUTPUTS/manifest.json" \
    --archive-out "$OUTPUTS/component.tar.zst"
printf 'local_component=pass\noutput_manifest=%s\n' "$OUTPUTS/manifest.json"
