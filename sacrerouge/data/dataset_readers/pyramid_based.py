from typing import List

from sacrerouge.data import EvalInstance, Pyramid, PyramidAnnotation
from sacrerouge.data.dataset_readers import DatasetReader
from sacrerouge.data.fields import Fields, PyramidField, PyramidAnnotationField
from sacrerouge.io import JsonlReader


@DatasetReader.register('pyramid-based')
class PyramidBasedDatasetReader(DatasetReader):
    def __init__(self, pyramid_jsonl: str, annotation_jsonl: str) -> None:
        super().__init__()
        self.pyramid_jsonl = pyramid_jsonl
        self.annotation_jsonl = annotation_jsonl

    def read(self) -> List[EvalInstance]:
        pyramids = JsonlReader(self.pyramid_jsonl, Pyramid).read()
        annotations = JsonlReader(self.annotation_jsonl, PyramidAnnotation).read()

        # Enumerate the peers
        instance_id_to_pyramid = {pyramid.instance_id: pyramid for pyramid in pyramids}
        eval_instances = []
        for annotation in annotations:
            summary = PyramidAnnotationField(annotation)
            pyramid = instance_id_to_pyramid[annotation.instance_id]
            fields = Fields({
                'pyramid': PyramidField(pyramid)
            })

            eval_instances.append(EvalInstance(
                annotation.instance_id,
                annotation.summarizer_id,
                annotation.summarizer_type,
                summary,
                fields
            ))

        # Enumerate the references
        for pyramid in pyramids:
            for i, summary in enumerate(pyramid.summaries):
                annotation = pyramid.get_annotation(i)
                summary = PyramidAnnotationField(annotation)

                fields = Fields({
                    'pyramid': PyramidField(pyramid.remove_summary(i))
                })

                eval_instances.append(EvalInstance(
                    annotation.instance_id,
                    annotation.summarizer_id,
                    annotation.summarizer_type,
                    summary,
                    fields
                ))

        return eval_instances
