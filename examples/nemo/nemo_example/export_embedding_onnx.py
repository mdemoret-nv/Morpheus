import onnx
import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers import BertModel


class CustomTokenizer(torch.nn.Module):

    def __init__(self, model_name: str):
        super().__init__()

        self.inner_model = AutoModel.from_pretrained(model_name)

        self._output_dim = self.inner_model.config.hidden_size

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        # Adapted from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        #First element of model_output contains all token embeddings
        token_embeddings = model_output[0]  # [batch_size, seq_length, hidden_size]

        # Transpose to make broadcasting possible
        token_embeddings = torch.transpose(token_embeddings, 0, 2)  # [hidden_size, seq_length, batch_size]

        input_mask_expanded = torch.transpose(attention_mask.unsqueeze(-1).float(), 0, 2)  # [1, seq_length, batch_size]

        num = torch.sum(token_embeddings * input_mask_expanded, 1)  # [hidden_size, batch_size]
        denom = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # [1, batch_size]

        return torch.transpose(num / denom, 0, 1)  # [batch_size, hidden_size]

    def normalize(self, embeddings):

        # Use the same trick here to broadcast to avoid using the expand operator which breaks dynamic axes
        denom = torch.transpose(embeddings.norm(2, 1, keepdim=True).clamp_min(1e-12), 0, 1)

        return torch.transpose(torch.transpose(embeddings, 0, 1) / denom, 0, 1)

    def forward(self, input_ids, attention_mask):
        model_outputs = self.inner_model(input_ids=input_ids, attention_mask=attention_mask)

        sentence_embeddings = self.mean_pooling(model_outputs, attention_mask)

        sentence_embeddings = self.normalize(sentence_embeddings)

        return sentence_embeddings


device = torch.device("cuda")

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# model_name = 'sentence-transformers/all-mpnet-base-v2'

# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = CustomTokenizer(model_name)
model.to(device)
model.eval()

batch_size = 10
max_seq_length = 512

input_ids = torch.ones(batch_size, max_seq_length, dtype=torch.int32).to(device)
#input_type_ids = torch.ones(batch_size, max_seq_length, dtype=torch.long).to(device)
input_mask = torch.ones(batch_size, max_seq_length, dtype=torch.int32).to(device)
#input_features = {'input_ids': input_ids, 'token_type_ids': input_type_ids, 'attention_mask': input_mask}
input_features = {'input_ids': input_ids, 'attention_mask': input_mask}

test_output = model(**input_features)

onnx_path = model_name

if (onnx_path.startswith("sentence-transformers/")):
    onnx_path = onnx_path.removeprefix("sentence-transformers/")

onnx_path = f"{onnx_path}.onnx"

torch.onnx.export(
    model,
    (input_ids, input_mask),
    onnx_path,
    opset_version=13,
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {
            0: 'batch_size',
            1: "seq_length",
        },  # variable length axes
        'attention_mask': {
            0: 'batch_size',
            1: "seq_length",
        },
        'output': {
            0: 'batch_size',
        }
    },
    verbose=True)

onnx_model = onnx.load(onnx_path)

onnx.checker.check_model(onnx_model)
