### PALM classification

Classification implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a>.

We add dropout to transformer layers and attention mask in TransformerBlock to shield out the uncessary inputs. We add classification in [AutoClassifierWrapper](../modeling_palm.py).

The whole idea of PaLm use Multi-Query Attention and Rotary Position Embedding, or RoPE in the architecture.

### Implementation

We try two different ways to implementation of answer smmarization with yes or no. The first one, we fine tuining the bert model as we can see from the file see documentation [bert_train](bert_train.ipynb). The second one, we implement the PaLm on binary classfication and train on boolq dataset [boolq_dataset](../../datasets/boolq/train.jsonl). We read from jsonl file, use adam optimizer and CrossEntropyLoss loss function.