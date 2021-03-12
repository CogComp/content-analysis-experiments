"""Reformats the summaries into a common format"""
import argparse
import json
import os
import zipfile


def _load_see2017(input_file: str):
    articles = {}
    references = {}
    outputs = {}

    with zipfile.ZipFile(input_file, 'r') as f:
        for name in f.namelist():
            if name.startswith('test_output/articles/') and name.endswith('.txt'):
                filename = os.path.basename(name)
                instance_id = filename[:6]
                text = f.read(name).decode()
                articles[instance_id] = text
            elif name.startswith('test_output/reference/') and name.endswith('.txt'):
                filename = os.path.basename(name)
                instance_id = filename[:6]
                text = f.read(name).decode().splitlines()
                references[instance_id] = text
            elif name.startswith('test_output/pointer-gen-cov/') and name.endswith('.txt'):
                filename = os.path.basename(name)
                instance_id = filename[:6]
                text = f.read(name).decode().splitlines()
                outputs[instance_id] = text

    instances = []
    for instance_id in outputs.keys():
        instances.append({
            'instance_id': f'see2017-{instance_id}',
            'document': articles[instance_id],
            'reference': references[instance_id],
            'summary': outputs[instance_id]
        })
    return instances


def _load_liu2019(input_file: str):
    instances = []

    with zipfile.ZipFile(input_file, 'r') as f:
        for i, line in enumerate(f.read('CNNDM_BertSumExtAbs.jsonl').decode().splitlines()):
            data = json.loads(line)
            article = data['article']
            summary = data['decoded'].split(' <q> ')
            reference = data['reference'].split(' <q> ')

            # The sentences don't have any punctuation on the ends, so add
            # a period to each
            for j, sentence in enumerate(summary):
                summary[j] = sentence + ' .'
            for j, sentence in enumerate(reference):
                reference[j] = sentence + ' .'

            instances.append({
                'instance_id': f'liu2019-{i}',
                'document': article,
                'reference': reference,
                'summary': summary
            })
    return instances


def _save(instances, file_path: str, name):
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(file_path, 'w') as out:
        for instance in instances:
            data = {
                'instance_id': instance['instance_id'],
                'summarizer_id': name,
                'summarizer_type': 'peer',
                'summary': {'text': instance['summary']},
                'references': [
                    {'text': instance['reference']}
                ]
            }
            out.write(json.dumps(data) + '\n')


def main(args):
    see2017 = _load_see2017(args.see_zip)
    liu2019 = _load_liu2019(args.liu_zip)

    _save(see2017, args.see_output_jsonl, 'see2017')
    _save(liu2019, args.liu_output_jsonl, 'liu2019')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--liu-zip', required=True)
    argp.add_argument('--see-zip', required=True)
    argp.add_argument('--liu-output-jsonl', required=True)
    argp.add_argument('--see-output-jsonl', required=True)
    args = argp.parse_args()
    main(args)