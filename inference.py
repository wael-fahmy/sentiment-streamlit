#pip install sagemaker
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

# Update with your execution role ARN (not your user ARN)
role = "arn:aws:iam::118169265223:role/sage"

# Model S3 path
model_data = "s3://cloud-computing-ai-inference/models/sentiment/model.tar.gz"


huggingface_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
  transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    sagemaker_session=sagemaker.Session()
)
# Deploy the model to SageMaker endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="ai-sentiment-endpoint"
)
