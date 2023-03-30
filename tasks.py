from datasets import classification_dataset_builder
from transforms import classification_simple_pipeline_builder

TASKS = {
    "classification-simple": (
        classification_simple_pipeline_builder,
        classification_dataset_builder,
    ),
}


def make_task(name, *, input_type, api_version, dataset_rng, num_samples):
    pipeline_builder, dataset_builder = TASKS[name]

    pipeline = pipeline_builder(input_type=input_type, api_version=api_version)
    if pipeline is None:
        return None

    dataset = dataset_builder(
        input_type=input_type,
        api_version=api_version,
        rng=dataset_rng,
        num_samples=num_samples,
    )

    return pipeline, dataset
