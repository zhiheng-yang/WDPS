### PALM classification

Classification implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a>.

Take credit from <a href="https://github.com/lucidrains/PaLM-pytorch/tree/main">PaLM-pytorch</a>, we add dropout to transformer layers and attention mask in TransformerBlock to shield out the uncessary inputs. We add classification in AutoClassifierWrapper.

The whole idea of PaLm use Multi-Query Attention and Rotary Position Embedding, or RoPE in the architecture.


