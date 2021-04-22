# !/bin/bash
# bash show_result.sh "/DATA/disk1/caiyueliang/wenet/examples/qudian/s1_cyl/exp/conformer" "67.pt"

cur_dir=$(cd "$(dirname "$0")"; pwd)

result_dir=${1}
model_name=${2}
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
label_file="/DATA/disk1/caiyueliang/QASR/asr_cer/transform_qd/test_filt.chars.txt"

for mode in ${decode_modes}; do
    echo "==============================================================="
    input_file=${result_dir}"/test_"${mode}"/text_"${model_name}
    python ${cur_dir}/trans_cer_text.py --input_file ${input_file} || exit 1
    bash ${cur_dir}/show_cer.sh ${label_file} ${input_file}"_trans" || exit 1
done

