DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

python ${DIR}/download.py \
  --liu-output-zip ${DIR}/liu2019.zip \
  --see-output-zip ${DIR}/see2017.zip

python ${DIR}/reformat.py \
  --liu-zip ${DIR}/liu2019.zip \
  --see-zip ${DIR}/see2017.zip \
  --liu-output-jsonl ${DIR}/liu2019.jsonl \
  --see-output-jsonl ${DIR}/see2017.jsonl