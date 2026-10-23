#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/model_convert"

WORK_TMP="${WORK_TMP:-${ROOT}/.work_tmp}"
TMPDIR="${TMPDIR:-${WORK_TMP}/tmp}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${WORK_TMP}/matplotlib}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_TMP}/xdg_cache}"
export TMPDIR MPLCONFIGDIR XDG_CACHE_HOME
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if ! command -v pulsar2 >/dev/null 2>&1; then
    set +u
    source "${NPU_DEV_ENV:-/data/huyuan/npu-codebase/script/npu_dev}"
    set -u
fi

MODEL_DIR="${MODEL_DIR:-${ROOT}/models/decoder4_static}"
CALIB_DIR="${CALIB_DIR:-${ROOT}/model_convert/calib_data_decoder4_rich}"
AXMODEL_DIR="${AXMODEL_DIR:-${ROOT}/model_convert/axmodels_decoder4_rich_check0}"
MAX_TOKENS="${MAX_TOKENS:-384}"
MAX_FEAT_LEN="${MAX_FEAT_LEN:-1024}"
FEAT_DIM="${FEAT_DIM:-100}"
DECODER_BATCH="${DECODER_BATCH:-2}"
MASK_INPUT_NAME="${MASK_INPUT_NAME:-padding_mask2}"
MASK_SHAPE="${MASK_SHAPE:-${DECODER_BATCH}x${MAX_FEAT_LEN}}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:--1}"
CHECK_LEVEL="${CHECK_LEVEL:-0}"
DEFAULT_DATA_TYPE="${DEFAULT_DATA_TYPE:-U16}"
BUILD_SUFFIX="${BUILD_SUFFIX:-}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
MIXED_U16_OP_TYPES="${MIXED_U16_OP_TYPES:-}"
PIECES="${PIECES:-fm_decoder_part0,fm_decoder_part1,fm_decoder_part2,fm_decoder_part3}"
TARGET_HARDWARE="${TARGET_HARDWARE:-AX650}"
NPU_MODE="${NPU_MODE:-NPU3}"

mkdir -p "${AXMODEL_DIR}"

should_run_piece() {
    local name="$1"
    IFS=',' read -ra pieces <<< "${PIECES}"
    for piece in "${pieces[@]}"; do
        if [ "${piece}" = "${name}" ]; then
            return 0
        fi
    done
    return 1
}

write_common_quant_config() {
    local mixed_configs=""
    if [ -n "${MIXED_U16_OP_TYPES}" ]; then
        IFS=',' read -ra mixed_ops <<< "${MIXED_U16_OP_TYPES}"
        for op_type in "${mixed_ops[@]}"; do
            mixed_configs="${mixed_configs}            {\"op_type\": \"${op_type}\", \"data_type\": \"U16\"},\n"
        done
    fi
    cat << EOF
        "layer_configs": [
            {"op_type": "Pow", "data_type": "U8"},
$(printf "%b" "${mixed_configs}")            {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "${DEFAULT_DATA_TYPE}"}
        ],
        "calibration_method": "MinMax",
        "enable_smooth_quant": true,
        "conv_bias_data_type": "FP32",
        "precision_analysis": true,
        "precision_analysis_method": "EndToEnd",
        "disable_auto_refine_scale": true,
        "transformer_opt_level": 0
EOF
}

write_input_configs() {
    local part="$1"
    shift
    local first=1
    for tensor in "$@"; do
        if [ "${first}" -eq 0 ]; then
            echo ","
        fi
        first=0
        printf '            {"tensor_name": "%s", "calibration_dataset": "%s/%s/%s.tar.gz", "calibration_format": "Numpy", "calibration_size": %s}' \
            "${tensor}" "${CALIB_DIR}" "${part}" "${tensor}" "${CALIBRATION_SIZE}"
    done
    echo
}

build_decoder_piece() {
    local part="$1"
    local output_name="$2"
    local input_shapes="$3"
    shift 3
    local tensors=("$@")
    local config="config_${part}_check0${BUILD_SUFFIX}.json"
    local build_dir="build-${part}-check0${BUILD_SUFFIX}"
    local output_file="${output_name%.axmodel}${OUTPUT_SUFFIX}.axmodel"
    case "${part}" in
        fm_decoder_part0)
            output_file="${OUTPUT_NAME_PART0:-${output_file}}"
            ;;
        fm_decoder_part1)
            output_file="${OUTPUT_NAME_PART1:-${output_file}}"
            ;;
        fm_decoder_part2)
            output_file="${OUTPUT_NAME_PART2:-${output_file}}"
            ;;
        fm_decoder_part3)
            output_file="${OUTPUT_NAME_PART3:-${output_file}}"
            ;;
    esac
    cat > "${config}" << EOF
{
    "model_type": "ONNX",
    "npu_mode": "${NPU_MODE}",
    "input": "${MODEL_DIR}/${part}.onnx",
    "output_name": "${output_file}",
    "output_dir": "./${build_dir}",
    "target_hardware": "${TARGET_HARDWARE}",
    "onnx_opt": {"disable_onnx_optimization": false, "enable_onnxsim": true},
    "quant": {
        "input_configs": [
$(write_input_configs "${part}" "${tensors[@]}")
        ],
$(write_common_quant_config)
    },
    "input_processors": [{"tensor_name": "DEFAULT"}],
    "compiler": {"check": ${CHECK_LEVEL}, "enable_slice_mode": false}
}
EOF
    pulsar2 build --config "${config}" --input_shapes "${input_shapes}" 2>&1 | tee "${build_dir}.log"
    cp "${build_dir}/${output_file}" "${AXMODEL_DIR}/"
}

should_run_piece "fm_decoder_part0" && build_decoder_piece "fm_decoder_part0" "decoder_part0_check0.axmodel" "t:1;x:1x${MAX_FEAT_LEN}x${FEAT_DIM};text_condition:1x${MAX_FEAT_LEN}x${FEAT_DIM};speech_condition:1x${MAX_FEAT_LEN}x${FEAT_DIM};guidance_scale:1;padding_mask:1x${MAX_FEAT_LEN}" "t" "x" "text_condition" "speech_condition" "guidance_scale" "padding_mask" || echo "Skipping fm_decoder_part0"
should_run_piece "fm_decoder_part1" && build_decoder_piece "fm_decoder_part1" "decoder_part1_check0.axmodel" "decoder_hidden_p0:${MAX_FEAT_LEN}x${DECODER_BATCH}x512;time_hidden:${DECODER_BATCH}x192;${MASK_INPUT_NAME}:${MASK_SHAPE}" "decoder_hidden_p0" "time_hidden" "${MASK_INPUT_NAME}" || echo "Skipping fm_decoder_part1"
should_run_piece "fm_decoder_part2" && build_decoder_piece "fm_decoder_part2" "decoder_part2_check0.axmodel" "decoder_hidden_p1:${MAX_FEAT_LEN}x${DECODER_BATCH}x512;time_hidden:${DECODER_BATCH}x192;${MASK_INPUT_NAME}:${MASK_SHAPE}" "decoder_hidden_p1" "time_hidden" "${MASK_INPUT_NAME}" || echo "Skipping fm_decoder_part2"
if [ "${DECODER_BATCH}" = "1" ]; then
    should_run_piece "fm_decoder_part3" && build_decoder_piece "fm_decoder_part3" "decoder_part3_check0.axmodel" "decoder_hidden_p2:${MAX_FEAT_LEN}x${DECODER_BATCH}x512;time_hidden:${DECODER_BATCH}x192;${MASK_INPUT_NAME}:${MASK_SHAPE}" "decoder_hidden_p2" "time_hidden" "${MASK_INPUT_NAME}" || echo "Skipping fm_decoder_part3"
else
    should_run_piece "fm_decoder_part3" && build_decoder_piece "fm_decoder_part3" "decoder_part3_check0.axmodel" "decoder_hidden_p2:${MAX_FEAT_LEN}x${DECODER_BATCH}x512;time_hidden:${DECODER_BATCH}x192;${MASK_INPUT_NAME}:${MASK_SHAPE};cfg_scale:1" "decoder_hidden_p2" "time_hidden" "${MASK_INPUT_NAME}" "cfg_scale" || echo "Skipping fm_decoder_part3"
fi

ls -lh "${AXMODEL_DIR}"/*.axmodel
