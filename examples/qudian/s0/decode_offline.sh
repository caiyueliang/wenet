#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

export NCCL_DEBUG=INFO
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

wav_dir="/DATA/disk1/ASR/qd_2_aishell/data_aishell/wav/test"
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
    # Data preparation
    test_dir=data/local/${test_set}
    mkdir -p ${test_dir}
    find ${wav_dir} -iname "*.wav" > ${test_dir}/wav.flist

    # Transcriptions preparation
    echo "Preparing ${test_dir} transcriptions"
    sed -e 's/\.wav//' ${test_dir}/wav.flist | awk -F '/' '{print $NF}' > ${test_dir}/utt.list
    paste -d' ' ${test_dir}/utt.list ${test_dir}/wav.flist > ${test_dir}/wav.scp_all
    tools/filter_scp.pl -f 1 ${test_dir}/utt.list ${test_dir}/wav.scp_all | sort -u > ${test_dir}/wav.scp

    mkdir -p data/${test_set}
    for f in wav.scp; do
      cp ${test_dir}/${f} data/${test_set}/${f} || exit 1;
    done
    # mkdir -p ${feat_dir}
    # cp -r data/${test_set} ${feat_dir}
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
    python wenet/bin/decode.py --gpu 0 \
        --mode ${decode_mode} \
        --config ${dir}/train.yaml \
        --test_data data/${test_set}/wav.scp \
        --checkpoint ${decode_checkpoint} \
        --beam_size 10 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict ${dict} \
        --ctc_weight ${ctc_weight} \
        --result_file ${test_dir}/text_${model_name} \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} || exit 1;
fi

decode_time=`date +'%Y-%m-%d %H:%M:%S'`
echo "[decode_time] "${decode_time}
time=$(($(date +%s -d "${decode_time}") - $(date +%s -d "${start_time}")));
echo " [total_used] "${time}" s"

