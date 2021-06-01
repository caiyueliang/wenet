#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="0,1"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

wav_dir="/DATA/disk1/ASR/qd_data_all/corpus/test"
nj=16
# feat_dir=raw_wav
dict=data/dict/lang_char.txt

train_config=conf/train_conformer.yaml
dir=exp/conformer

decode_checkpoint=${dir}/final.pt
decode_mode="attention_rescoring"
test_set="offline_test"

. tools/parse_options.sh || exit 1;

start_time=`date +'%Y-%m-%d %H:%M:%S'`
echo "[start_time] "${start_time}

for d in "data/local" "data/${test_set}" "raw_wav"; do
    if [[ -d ${d} ]];then
        rm -rf ${d}
    fi
done

if [[ ${stage} -le 1 ]] && [[ ${stop_stage} -ge 1 ]]; then
    echo "[stage:1] ================================================="
    python tools/gen_wav_scp.py \
        --wav_dir ${wav_dir} \
        --split ${nj} \
        --out_dir data/${test_set} || exit 1;
fi

if [[ ${stage} -le 2 ]] && [[ ${stop_stage} -ge 2 ]]; then
    echo "[stage:2] ================================================="
    # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
    # -1 for full chunk
    decoding_chunk_size=
    ctc_weight=0.5

    test_dir=${dir}/test_${decode_mode}
    mkdir -p ${test_dir}
    model_name=${decode_checkpoint##*/}
    gpu_id=$(echo ${CUDA_VISIBLE_DEVICES} | cut -d',' -f$[1])
    echo "[test] mode: ${decode_mode}, use gpu_id: ${gpu_id}, model_name: ${model_name}"
    for ((i = 0; i < ${nj}; ++i)); do
    {
        python wenet/bin/decode.py --gpu ${gpu_id} \
            --mode ${decode_mode} \
            --config ${dir}/train.yaml \
            --test_data data/${test_set}/wav.${i}.scp \
            --checkpoint ${decode_checkpoint} \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict ${dict} \
            --ctc_weight ${ctc_weight} \
            --result_file ${test_dir}/text_${model_name}.${i}.log \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    } &
    done
    wait

    cat ${test_dir}/text_${model_name}.*.log > ${test_dir}/text_${model_name}.txt
fi

decode_time=`date +'%Y-%m-%d %H:%M:%S'`
echo "[decode_time] "${decode_time}
time=$(($(date +%s -d "${decode_time}") - $(date +%s -d "${start_time}")));
echo " [total_used] "${time}" s"

