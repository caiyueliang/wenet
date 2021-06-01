#!/bin/bash
compute_wer="/DATA/disk1/guojie/kaldi/src/bin/compute-wer"
label_file=${1}
predict_file=${2}

echo "  [label_file] "${label_file}
echo "[predict_file] "${predict_file}

echo "---------------------------------------------------------------"
echo "[total] cer:"
cat ${predict_file} | ${compute_wer} --text --mode=present ark:${label_file}  ark,p:-

echo "---------------------------------------------------------------"
for y in seat user call debt; do
    echo " >>>>>> "${y}" cer:"
    grep ${y} ${predict_file} | ${compute_wer} --text --mode=present ark:${label_file}  ark,p:-
done

echo "---------------------------------------------------------------"
for y in  call debt; do
    for x in seat user; do
        echo " >>>>>> ${y}:${x} cer:"
        grep ${y} ${predict_file} | grep ${x} | ${compute_wer} --text --mode=present ark:${label_file}  ark,p:-
    done
done
