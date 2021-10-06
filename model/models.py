import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.autograd import Variable

from transformers import T5EncoderModel

from transformers.modeling_outputs import SequenceClassifierOutput

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_p)

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


class FocalLoss(nn.Module):

    # Implementation: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,torch.LongTensor)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class CustomT5EncoderForSequenceClassificationMean(T5EncoderModel):
    def __init__(self, config):

        super(CustomT5EncoderForSequenceClassificationMean, self).__init__(config)

        self.num_labels = config.num_labels
        self.focal_loss = True if config.focal_loss else False

        self.loss_fn = None
        if self.focal_loss:
            self.loss_fn = FocalLoss(gamma=0.5)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1] + [1]*29))

        self.pooler = MeanPooler(config)
        self.dropout = nn.Dropout(config.dropout_p)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GRUT5EncoderForSequenceClassification(T5EncoderModel):

    def __init__(self, config):

        super(GRUT5EncoderForSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.focal_loss = True if config.focal_loss else False

        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 
                          num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(config.dropout_p)
        self.fc1 = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_labels)

        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1] + [1]*29))


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

        _, h_n = self.gru(last_hidden_state)
        concat = torch.cat([h_n[-1], h_n[-2]], dim=-1)
        concat = self.dropout(concat)
        concat = self.fc1(concat)
        concat = self.dropout(concat)
        logits = self.fc2(concat)

        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FrozenT5EncoderForSequenceClassificationMean(CustomT5EncoderForSequenceClassificationMean):
    def __init__(self, config):
        super(FrozenT5EncoderForSequenceClassificationMean, self).__init__(config)
        for param in self.encoder.parameters():
            param.requires_grad = False


class T5EncoderForSequenceClassificationMeanSubmeanObjmean(T5EncoderModel):
    def __init__(self, config):
        
        super(T5EncoderForSequenceClassificationMeanSubmeanObjmean, self).__init__(config)
        self.num_labels = config.num_labels        
        self.model_dim = config.d_model
        
        self.pooler = MeanPooler(config)
        self.dropout = nn.Dropout(config.dropout_p)
        self.fc_layer = nn.Sequential(nn.Linear(self.model_dim, self.model_dim))
        self.classifier = nn.Sequential(nn.Linear(self.model_dim*3, self.num_labels))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        entity_token_idx=None
    ):

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

        pooled_output = self.pooler(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)

        sub_output = []
        obj_output = []

        # b_idx: batch index
        # entity_idx = [sub_entity_idx, obj_entity_idx]
        # sub_entity_idx = [start, end]
        # obj_entity_idx = [start, end]

        for b_idx, entity_idx in enumerate(entity_token_idx):
           
            sub_entity_idx, obj_entity_idx = entity_idx

            sub_hidden = last_hidden_state[b_idx, sub_entity_idx[0]:sub_entity_idx[1]+1, :]
            sub_hidden_mean = torch.mean(sub_hidden, 0)
            sub_output.append(sub_hidden_mean.unsqueeze(0))
            
            obj_hidden = last_hidden_state[b_idx, obj_entity_idx[0]:obj_entity_idx[1]+1, :]
            obj_hidden_mean = torch.mean(obj_hidden, 0)
            obj_output.append(obj_hidden_mean.unsqueeze(0))

            if (sub_entity_idx[0] == sub_entity_idx[1]):
                print("SUB ERROR!!!!!!!!!")
                print(sub_entity_idx)
                print(sub_hidden_mean)

            elif (obj_entity_idx[0] == obj_entity_idx[1]):
                print("OBJ ERROR!!!!!!!!!")
                print(obj_entity_idx)
                print(obj_hidden_mean)
            
        sub_hidden_mean_cat = self.fc_layer(torch.cat((sub_output)))
        obj_hidden_mean_cat = self.fc_layer(torch.cat((obj_output)))

        entities_concat = torch.cat([pooled_output, sub_hidden_mean_cat, obj_hidden_mean_cat], dim=-1)
        entities_concat = self.dropout(entities_concat)
        
        logits = self.classifier(entities_concat)
        
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )