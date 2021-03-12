DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

for dataset in tac2008 tac2009; do
  for metric in rouge bertscore; do
    python -m sacrerouge score \
      ${DIR}/configs/${metric}.json \
      ${DIR}/output/${dataset}/${metric}/scores.jsonl \
      --overrides '{"dataset_reader.pyramid_jsonl": "datasets/tac/'"${dataset}"'.pyramids.jsonl", "dataset_reader.annotation_jsonl": "datasets/tac/'"${dataset}"'.pyramid-annotations.jsonl"}'

    python ${DIR}/analyze.py \
      ${metric} \
      ${DIR}/output/${dataset}/${metric}/scores.jsonl \
      peer \
      ${DIR}/output/${dataset}/${metric}/plots/contribution.peers.pdf \
      ${DIR}/output/${dataset}/${metric}/plots/coverage.peers.pdf \
      ${DIR}/output/${dataset}/${metric}/stats.json
    done
done