import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MeanPooler(nn.Module):
    def __init__(self, embedding_dims: int = 1024, dropout_p: float = 0.5):
        super().__init__()
        self.dense1 = nn.Linear(embedding_dims, embedding_dims)
        self.dense2 = nn.Linear(embedding_dims, embedding_dims)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, hidden_states, attention_mask, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]

        sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), attention_mask.float().unsqueeze(-1)).squeeze(-1)
        divisor = attention_mask.sum(dim=1).view(-1, 1).float()

        if sqrt:
            divisor = divisor.sqrt()
        sentence_sums /= divisor

        pooled_output = self.dense1(sentence_sums)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)

        return pooled_output


class CustomT5EncoderForSequenceClassificationMean(nn.Module):
    def __init__(self,
                 num_labels: int = 30,
                 embedding_dims: int = 1024,
                 dropout_p: float = 0.5, 
                 model_name: str = 'KETI-AIR/ke-t5-large',
                 load_model: str = None):

        super(CustomT5EncoderForSequenceClassificationMean, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        self.num_labels = num_labels
        if model_name == 'KETI-AIR/ke-t5-large':
            assert embedding_dims == 1024
        elif model_name == 'KETI-AIR/ke-t5-base':
            assert embedding_dims == 768
        elif model_name == 'KETI-AIR/ke-t5-small':
            assert embedding_dims == 512

        self.pooler = MeanPooler(embedding_dims=embedding_dims, dropout_p=dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(embedding_dims, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = True

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs[0]
        print(last_hidden_state.shape)

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
