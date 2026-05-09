#!/bin/bash

sed -n -f <(awk '{print $1"p"}' line-no.txt) /mnt/disk1/ye/exp/myMono_LFM/syl/myMono_corpus.ver.0.1.shuf.syl > test2.txt

