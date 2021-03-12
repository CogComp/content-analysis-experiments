DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

for dataset in tac2008 tac2009; do
  for metric in rouge bertscore; do
    python -m sacrerouge score \
      ${DIR}/configs/${metric}.json \
      ${DIR}/output/${dataset}/${metric}/scores/interpretable.jsonl \
      --overrides '{"dataset_reader.input_jsonl": "datasets/tac/'"${dataset}"'.summaries.jsonl"}'

    for summarizer_type in 'all' 'peer'; do
      python ${DIR}/../category_contributions.py \
        ${DIR}/output/${dataset}/${metric}/scores/interpretable.jsonl \
        ${summarizer_type} \
        ${DIR}/output/tables/${dataset}/${metric}/${summarizer_type}/category-coverages.tex \
        ${DIR}/output/tables/${dataset}/${metric}/${summarizer_type}/content-type-coverages.tex
    done
  done
done
