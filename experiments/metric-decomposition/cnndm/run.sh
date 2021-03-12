DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

for metric in rouge bertscore; do
  for system in see2017 liu2019; do
    python -m sacrerouge evaluate \
      ${DIR}/configs/${metric}.json \
      ${DIR}/output/${metric}/scores/${system}.macro.json \
      ${DIR}/output/${metric}/scores/${system}.micro.jsonl \
      --overrides '{"dataset_reader.input_jsonl": "datasets/cnndm/'"${system}"'.jsonl"}' \
      --silent
  done

  temp_file=$(mktemp)
  for system in see2017 liu2019; do
    cat ${DIR}/output/${metric}/scores/${system}.micro.jsonl >> ${temp_file}
  done

  python ${DIR}/../category_contributions.py \
    ${temp_file} \
    peer \
    ${DIR}/output/tables/${metric}/category-coverages.tex \
    ${DIR}/output/tables/${metric}/content-type-coverages.tex
  rm ${temp_file}

  python ${DIR}/system_diff.py \
    ${DIR}/output/${metric}/scores/see2017.macro.json \
    ${DIR}/output/${metric}/scores/liu2019.macro.json \
    ${DIR}/output/${metric}/diff.tex
done
