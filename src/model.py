from transformers import BertConfig
from bert_model import  BertModel
import sys
from mindspore import nn,Parameter,ops
from mindspore.ops import stop_gradient
import mindspore
mindspore.set_context(device_target='CPU',device_id=1)
class LOTClassModel(nn.Cell):

    def __init__(self,num_class):
        super(LOTClassModel,self).__init__()
        config = BertConfig.from_pretrained('../bert-base-uncased')
        config.dtype  =mindspore.dtype.float32
        config.compute_type = mindspore.dtype.float16
        self.num_labels = num_class

        self.bert = BertModel(config,is_training=True,use_one_hot_embeddings=False)
        #self.bert = RobertaModel(config, add_pooling_layer=True)
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size)

        # self.bias = Parameter(ops.zeros(self.config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # self.decoder.bias = self.bias
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Dense(config.hidden_size, num_class)
        #self.init_weights()
        # MLM head is not trained
        # for param in self.dense.get_parameters():
        #     param.requires_grad = False
        #self.dense = stop_gradient(self.dense)
    
    def construct(self, input_ids, pred_mode, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        # input_ids = mindspore.Tensor(input_ids)
        # attention_mask = mindspore.Tensor(attention_mask)
        token_type_id = ops.zeros_like(attention_mask)
        bert_outputs = self.bert(input_ids,
                                 input_mask=attention_mask,
                                 token_type_ids=token_type_id
                            )
        last_hidden_states = bert_outputs[0]
        pool_out = bert_outputs[1]


        if pred_mode == "classification":
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            trans_states = self.dropout(trans_states)
            logits = self.classifier(trans_states)
        elif pred_mode == "mlm":
            logits = self.dense(last_hidden_states)
        #chenhu
        elif pred_mode == "mcp":
            logits = self.classifier(pool_out)
        else:
            sys.exit("Wrong pred_mode!")
        return logits
