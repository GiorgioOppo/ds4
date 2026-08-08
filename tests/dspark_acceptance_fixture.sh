#!/bin/sh
set -eu

DS4_BIN=${DS4_BIN:-./ds4}
MODEL=${DS4_DSPARK_MODEL:-${DS4_TEST_MODEL:-./ds4flash.gguf}}
SUPPORT=${DS4_DSPARK_SUPPORT:-gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
TOKENS=${DS4_DSPARK_FIXTURE_TOKENS:-32}
REQUIRE_PARTIAL=${DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL:-0}
REQUIRE_DIRECT=${DS4_DSPARK_FIXTURE_REQUIRE_DIRECT_COMMIT:-1}
REQUIRE_IDENTICAL=${DS4_DSPARK_FIXTURE_REQUIRE_IDENTICAL:-0}
PROPOSAL_QUALITY_GUARD=${DS4_DSPARK_FIXTURE_REQUIRE_PROPOSAL_QUALITY:-auto}
C_ADD_MIN_ACCEPTED=${DS4_DSPARK_FIXTURE_C_ADD_MIN_ACCEPTED:-8}
CONFIDENCE=${DS4_DSPARK_FIXTURE_CONFIDENCE:-}
TEMPERATURE=${DS4_DSPARK_FIXTURE_TEMPERATURE:-0}
TOP_P=${DS4_DSPARK_FIXTURE_TOP_P:-0.95}
MIN_P=${DS4_DSPARK_FIXTURE_MIN_P:-0.05}
SEED=${DS4_DSPARK_FIXTURE_SEED:-12345}
EXACT_SAMPLING=${DS4_DSPARK_FIXTURE_EXACT_SAMPLING:-0}
exact_sampling_arg=
if [ "$EXACT_SAMPLING" != 0 ]; then
    exact_sampling_arg=--mtp-exact-sampling
fi
partial_cases=0
direct_partial_cases=0
direct_commits=0
BACKEND=${DS4_DSPARK_FIXTURE_BACKEND:-auto}
SSD_STREAMING=${DS4_DSPARK_FIXTURE_SSD_STREAMING:-0}
SSD_CACHE_EXPERTS=${DS4_DSPARK_FIXTURE_SSD_STREAMING_CACHE_EXPERTS:-}
REQUIRE_ACTIVE=${DS4_DSPARK_FIXTURE_REQUIRE_ACTIVE:-1}
total_proposed=0
total_accepted_draft=0

case "$BACKEND" in
auto|metal|cuda|rocm) ;;
*)
    echo "dspark-fixture: invalid DS4_DSPARK_FIXTURE_BACKEND=$BACKEND" >&2
    exit 1
    ;;
esac
case "$SSD_STREAMING" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_SSD_STREAMING must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_ACTIVE" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_ACTIVE must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$SSD_CACHE_EXPERTS" in
""|*[!0-9]*)
    if [ -n "$SSD_CACHE_EXPERTS" ]; then
        echo "dspark-fixture: invalid SSD streaming expert count $SSD_CACHE_EXPERTS" >&2
        exit 1
    fi
    ;;
esac
if [ "$SSD_STREAMING" = 0 ] && [ -n "$SSD_CACHE_EXPERTS" ]; then
    echo "dspark-fixture: SSD cache experts requires DS4_DSPARK_FIXTURE_SSD_STREAMING=1" >&2
    exit 1
fi

proposal_quality_guard_enabled() {
    case "$PROPOSAL_QUALITY_GUARD" in
    0|false|no|off)
        return 1
        ;;
    1|true|yes|on)
        return 0
        ;;
    auto|"")
        [ "$REQUIRE_PARTIAL" = 0 ] || return 1
        [ -z "$CONFIDENCE" ] || return 1
        [ "$TOKENS" -ge 32 ] 2>/dev/null || return 1
        return 0
        ;;
    *)
        echo "dspark-fixture: invalid DS4_DSPARK_FIXTURE_REQUIRE_PROPOSAL_QUALITY=$PROPOSAL_QUALITY_GUARD" >&2
        exit 1
        ;;
    esac
}

case "$C_ADD_MIN_ACCEPTED" in
""|*[!0-9]*)
    echo "dspark-fixture: invalid DS4_DSPARK_FIXTURE_C_ADD_MIN_ACCEPTED=$C_ADD_MIN_ACCEPTED" >&2
    exit 1
    ;;
esac

PROPOSAL_QUALITY_GUARD_ACTIVE=0
if proposal_quality_guard_enabled; then
    PROPOSAL_QUALITY_GUARD_ACTIVE=1
fi

file_bytes() {
    if stat -L -f %z "$1" >/dev/null 2>&1; then
        stat -L -f %z "$1"
    elif stat -Lc %s "$1" >/dev/null 2>&1; then
        stat -Lc %s "$1"
    elif stat -f %z "$1" >/dev/null 2>&1; then
        stat -f %z "$1"
    elif stat -c %s "$1" >/dev/null 2>&1; then
        stat -c %s "$1"
    else
        echo unknown
    fi
}

git_commit_label() {
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    if [ "$commit" != unknown ] && ! git diff --quiet -- . 2>/dev/null; then
        commit="${commit}+dirty"
    fi
    echo "$commit"
}

print_metadata() {
    hw_os=$(uname -sm 2>/dev/null || echo unknown)
    hw_model=$(sysctl -n hw.model 2>/dev/null || true)
    hw_cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
    confidence=${CONFIDENCE:-default}

    printf '# commit=%s\n' "$(git_commit_label)"
    printf '# hardware_os=%s hardware_model=%s hardware_cpu=%s\n' \
        "$hw_os" "${hw_model:-unknown}" "${hw_cpu:-unknown}"
    printf '# model=%s model_bytes=%s support=%s support_bytes=%s\n' \
        "$MODEL" "$(file_bytes "$MODEL")" \
        "$SUPPORT" "$(file_bytes "$SUPPORT")"
    printf '# tokens=%s ctx=default flags="--temp %s --top-p %s --min-p %s --seed %s --nothink" exact_sampling=%s confidence=%s proposal_quality_guard=%s proposal_quality_active=%s c_add_min_accepted=%s require_direct=%s require_identical=%s\n' \
        "$TOKENS" "$TEMPERATURE" "$TOP_P" "$MIN_P" "$SEED" \
        "$EXACT_SAMPLING" "$confidence" "$PROPOSAL_QUALITY_GUARD" \
        "$PROPOSAL_QUALITY_GUARD_ACTIVE" "$C_ADD_MIN_ACCEPTED" \
        "$REQUIRE_DIRECT" "$REQUIRE_IDENTICAL"
    printf '# backend=%s ssd_streaming=%s ssd_cache_experts=%s require_active=%s\n' \
        "$BACKEND" "$SSD_STREAMING" "${SSD_CACHE_EXPERTS:-auto}" "$REQUIRE_ACTIVE"
    printf '# baseline_command=%s -m %s --tokens %s --temp %s --top-p %s --min-p %s --seed %s --nothink -p <fixture-prompt>\n' \
        "$DS4_BIN" "$MODEL" "$TOKENS" "$TEMPERATURE" "$TOP_P" "$MIN_P" "$SEED"
    printf '# dspark_command=DS4_DSPARK_STATS=1 %s --dspark%s%s -m %s --mtp-model %s --tokens %s --temp %s --top-p %s --min-p %s --seed %s --nothink -p <fixture-prompt>\n' \
        "$DS4_BIN" "${exact_sampling_arg:+ $exact_sampling_arg}" \
        "${CONFIDENCE:+ --dspark-confidence $CONFIDENCE}" \
        "$MODEL" "$SUPPORT" "$TOKENS" "$TEMPERATURE" "$TOP_P" "$MIN_P" "$SEED"
}

if [ ! -x "$DS4_BIN" ]; then
    echo "dspark-fixture: skipped, missing executable $DS4_BIN" >&2
    exit 0
fi
if [ ! -f "$MODEL" ]; then
    echo "dspark-fixture: skipped, missing model $MODEL" >&2
    exit 0
fi
if [ ! -f "$SUPPORT" ]; then
    echo "dspark-fixture: skipped, missing DSpark support model $SUPPORT" >&2
    exit 0
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dspark-fixture.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

run_variant() {
    mode=$1
    prompt=$2
    stdout_file=$3
    stderr_file=$4

    set -- "$DS4_BIN"
    case "$BACKEND" in
    metal) set -- "$@" --metal ;;
    cuda)  set -- "$@" --cuda ;;
    rocm)  set -- "$@" --rocm ;;
    esac
    if [ "$SSD_STREAMING" = 1 ]; then
        set -- "$@" --ssd-streaming
        if [ -n "$SSD_CACHE_EXPERTS" ]; then
            set -- "$@" --ssd-streaming-cache-experts "$SSD_CACHE_EXPERTS"
        fi
    fi
    if [ "$mode" = dspark ]; then
        set -- "$@" --dspark --mtp-model "$SUPPORT"
        if [ "$EXACT_SAMPLING" != 0 ]; then
            set -- "$@" --mtp-exact-sampling
        fi
        if [ -n "$CONFIDENCE" ]; then
            set -- "$@" --dspark-confidence "$CONFIDENCE"
        fi
    fi
    set -- "$@" -m "$MODEL" --tokens "$TOKENS" \
        --temp "$TEMPERATURE" --top-p "$TOP_P" --min-p "$MIN_P" \
        --seed "$SEED" --nothink -p "$prompt"

    if [ "$mode" = dspark ]; then
        DS4_DSPARK_STATS=1 "$@" >"$stdout_file" 2>"$stderr_file"
    else
        "$@" >"$stdout_file" 2>"$stderr_file"
    fi
}

run_case() {
    id=$1
    prompt=$2
    base_out="$tmpdir/$id.baseline.out"
    base_err="$tmpdir/$id.baseline.err"
    dspark_out="$tmpdir/$id.dspark.out"
    dspark_err="$tmpdir/$id.dspark.err"

    run_variant baseline "$prompt" "$base_out" "$base_err"
    run_variant dspark "$prompt" "$dspark_out" "$dspark_err"

    output_match=1
    if ! cmp -s "$base_out" "$dspark_out"; then
        output_match=0
        if [ "$REQUIRE_IDENTICAL" != 0 ]; then
            echo "dspark-fixture: output mismatch for $id" >&2
            echo "baseline:" >&2
            sed 's/^/  /' "$base_out" >&2
            echo "dspark:" >&2
            sed 's/^/  /' "$dspark_out" >&2
            return 1
        fi
    fi

    base_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$base_err" | tail -n 1)
    dspark_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$dspark_err" | tail -n 1)
    stats=$(grep 'DSpark stats' "$dspark_err" | tail -n 1 | sed 's/^ds4: DSpark stats //')
    if [ -z "$stats" ]; then
        echo "dspark-fixture: missing DSpark stats for $id" >&2
        return 1
    fi

    partial=$(printf '%s\n' "$stats" | sed -n 's/.* partial=\([0-9][0-9]*\).*/\1/p')
    errors=$(printf '%s\n' "$stats" | sed -n 's/.*errors=\([0-9][0-9]*\).*/\1/p')
    verifier_unavailable=$(printf '%s\n' "$stats" | sed -n 's/.*verifier_unavailable=\([0-9][0-9]*\).*/\1/p')
    proposed=$(printf '%s\n' "$stats" | sed -n 's/.*proposed=\([0-9][0-9]*\).*/\1/p')
    accepted_draft=$(printf '%s\n' "$stats" | sed -n 's/.*accepted_draft=\([0-9][0-9]*\).*/\1/p')
    direct_full=$(printf '%s\n' "$stats" | sed -n 's/.*direct_full=\([0-9][0-9]*\).*/\1/p')
    direct_partial=$(printf '%s\n' "$stats" | sed -n 's/.*direct_partial=\([0-9][0-9]*\).*/\1/p')
    partial=${partial:-0}
    errors=${errors:-0}
    verifier_unavailable=${verifier_unavailable:-0}
    proposed=${proposed:-0}
    accepted_draft=${accepted_draft:-0}
    direct_full=${direct_full:-0}
    direct_partial=${direct_partial:-0}
    if [ "$errors" -ne 0 ]; then
        echo "dspark-fixture: verifier errors for $id: $stats" >&2
        return 1
    fi
    if [ "$verifier_unavailable" -ne 0 ]; then
        echo "dspark-fixture: verifier unavailable for $id: $stats" >&2
        return 1
    fi
    total_proposed=$((total_proposed + proposed))
    total_accepted_draft=$((total_accepted_draft + accepted_draft))
    if [ "$PROPOSAL_QUALITY_GUARD_ACTIVE" -ne 0 ] && [ "$id" = c_add ] &&
        [ "$accepted_draft" -lt "$C_ADD_MIN_ACCEPTED" ]; then
        echo "dspark-fixture: c_add accepted_draft $accepted_draft below required $C_ADD_MIN_ACCEPTED: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_PARTIAL" != 0 ] && [ "$partial" -gt 0 ]; then
        partial_cases=$((partial_cases + 1))
    fi
    if [ "$direct_partial" -gt 0 ]; then
        direct_partial_cases=$((direct_partial_cases + 1))
    fi
    direct_commits=$((direct_commits + direct_full + direct_partial))

    printf '%s\toutput_match=%s\tbaseline_tps=%s\tdspark_tps=%s\t%s\n' \
        "$id" "$output_match" "${base_tps:-n/a}" "${dspark_tps:-n/a}" "$stats"
}

print_metadata
echo "id	output_match	baseline_tps	dspark_tps	dspark_stats"
run_case hello 'Hello'
run_case redis 'Explain Redis in one sentence.'
run_case math 'What is 17 times 23?'
run_case python_reverse 'Write a Python function that reverses a string.'
run_case c_add 'Complete this C function: int add(int a, int b) {'

if [ "$REQUIRE_PARTIAL" != 0 ] && [ "$partial_cases" -eq 0 ]; then
    echo "dspark-fixture: expected at least one partial accept case" >&2
    exit 1
fi
if [ "$REQUIRE_PARTIAL" != 0 ] && [ "$direct_partial_cases" -eq 0 ]; then
    echo "dspark-fixture: expected at least one direct partial commit" >&2
    exit 1
fi
if [ "$REQUIRE_DIRECT" != 0 ] && [ "$direct_commits" -eq 0 ]; then
    echo "dspark-fixture: expected at least one direct verifier-state commit" >&2
    exit 1
fi
if [ "$REQUIRE_ACTIVE" != 0 ] &&
   { [ "$total_proposed" -eq 0 ] || [ "$total_accepted_draft" -eq 0 ]; }; then
    echo "dspark-fixture: DSpark runtime was not active (proposed=$total_proposed accepted_draft=$total_accepted_draft)" >&2
    exit 1
fi
