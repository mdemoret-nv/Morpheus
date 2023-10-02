#!/bin/bash
set -e

root=${1:-shared-dir}
mkdir -p $root
cd $root
pushd .
#
mkdir -p llama2
cd llama2
if [ ! -e llama-2-13b-chat.Q4_K_M.gguf ]; then
    wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf
fi
popd
mkdir -p dataset/CFO
cd dataset/CFO
if [ ! -e Q1FY24-CFO-Commentary.pdf ]; then
    wget https://s201.q4cdn.com/141608511/files/doc_financials/2024/Q1FY24/Q1FY24-CFO-Commentary.pdf
fi
if [ ! -e Q2FY24-CFO-Commentary.pdf ]; then
    wget https://s201.q4cdn.com/141608511/files/doc_financials/2024/Q2FY24/Q2FY24-CFO-Commentary.pdf
fi
