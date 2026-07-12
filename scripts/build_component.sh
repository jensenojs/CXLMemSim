#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly PROFILE=${ROOT}/manifests/build-profile.json
readonly CONTRACT=${ROOT}/manifests/artifact-contract.json
readonly WORK=${ROOT}/.work/component
readonly BUILD=${WORK}/build
readonly GUEST_SOURCE=${WORK}/guest-source
readonly PAYLOAD=${WORK}/payload
readonly EVIDENCE=${WORK}/evidence
readonly BASELINE=4910c7cf2c813698952857f20988ac66dae7fe9d

cd "$ROOT"

readarray -t profile < <(python3 - "$PROFILE" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print(p["target_arch"])
print(p["compiler"]["cc"])
print(p["compiler"]["cxx"])
print(p["cmake"]["build_type"])
print(p["cmake"]["policy_default_cmp0091"])
print(p["cmake"]["parallel"])
for key, value in sorted(p["cmake"]["definitions"].items()):
    print(f"-D{key}={value}")
PY
)

[[ ${profile[0]} == amd64 ]]
[[ $(uname -m) == x86_64 ]]
readonly PROFILE_CC=${profile[1]}
readonly PROFILE_CXX=${profile[2]}
readonly BUILD_TYPE=${profile[3]}
readonly CMP0091=${profile[4]}
readonly PARALLEL=${profile[5]}
readonly CMAKE_DEFINITIONS=("${profile[@]:6}")

rm -rf "$WORK"
mkdir -p "$BUILD" "$GUEST_SOURCE" "$PAYLOAD/bin" "$PAYLOAD/guest" "$EVIDENCE"

env CC="$PROFILE_CC" CXX="$PROFILE_CXX" cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_POLICY_DEFAULT_CMP0091="$CMP0091" \
    "${CMAKE_DEFINITIONS[@]}" 2>&1 | tee "$EVIDENCE/cmake-configure.log"

cmake --build "$BUILD" --parallel "$PARALLEL" --target \
    cxlmemsim_server test_dcd_gfam test_rob test_mem_stall test_bandwidth_model \
    2>&1 | tee "$EVIDENCE/cmake-build.log"

ctest --test-dir "$BUILD" --output-on-failure 2>&1 | tee "$EVIDENCE/ctest.log"
"$BUILD/cxlmemsim_server" --help >"$EVIDENCE/server-help.stdout" \
    2>"$EVIDENCE/server-help.stderr"

git archive HEAD qemu_integration/guest_libcuda | tar -x -C "$GUEST_SOURCE"
readarray -t guest < <(python3 - "$PROFILE" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))["guest_shim"]
print(p["cc"])
print(" ".join(p["cflags"]))
for target in p["make_targets"]:
    print(target)
PY
)
readonly GUEST_CC=${guest[0]}
readonly GUEST_CFLAGS=${guest[1]}
readonly GUEST_TARGETS=("${guest[@]:2}")
readonly GUEST_DIR=${GUEST_SOURCE}/qemu_integration/guest_libcuda

make -C "$GUEST_DIR" CC="$GUEST_CC" CFLAGS="$GUEST_CFLAGS" "${GUEST_TARGETS[@]}" \
    2>&1 | tee "$EVIDENCE/guest-shim-build.log"

install -m 0755 "$BUILD/cxlmemsim_server" "$PAYLOAD/bin/cxlmemsim_server"
install -m 0755 "$GUEST_DIR/libcuda.so.1" "$PAYLOAD/guest/libcuda.so.1"
ln -s libcuda.so.1 "$PAYLOAD/guest/libcuda.so"

python3 scripts/component_artifact.py verify-profile --payload "$PAYLOAD" --profile "$PROFILE"
readelf -d "$PAYLOAD/bin/cxlmemsim_server" >"$EVIDENCE/server-readelf-dynamic.txt"
readelf -d "$PAYLOAD/guest/libcuda.so.1" >"$EVIDENCE/shim-readelf-dynamic.txt"
ldd "$PAYLOAD/bin/cxlmemsim_server" >"$EVIDENCE/server-ldd.txt"
ldd "$PAYLOAD/guest/libcuda.so.1" >"$EVIDENCE/shim-ldd.txt"
sha256sum "$PAYLOAD/bin/cxlmemsim_server" "$PAYLOAD/guest/libcuda.so.1" \
    >"$EVIDENCE/payload-sha256.txt"

printf 'component_build=pass\n'
printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
printf 'payload_dir=%s\n' "$PAYLOAD"
