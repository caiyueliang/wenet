export GLOG_logtostderr=1
export GLOG_v=2

wav_path=./20210327_unified_transformer_exp_server/BAC009S0764W0121.wav
model_dir=./20210327_unified_transformer_exp_server

../build/decoder_main \
    --chunk_size -1 \
    --wav_path ${wav_path} \
    --model_path ${model_dir}/final.zip \
    --dict_path ${model_dir}/words.txt 2>&1 | tee log.txt
