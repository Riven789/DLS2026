# Primary references

1. Vaswani, A. et al. (2017). **Attention Is All You Need.** NeurIPS. https://arxiv.org/abs/1706.03762. Attention equations, head concatenation, masking, and the original positionwise FFN. The original encoder-decoder model uses architectural choices that differ from this GPT-style baseline.
2. Radford, A. et al. (2019). **Language Models are Unsupervised Multitask Learners.** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf. Official implementation: https://github.com/openai/gpt-2/blob/master/src/model.py. Decoder-only causal architecture, learned position embeddings, and pre-normalization.
3. Sennrich, R., Haddow, B., and Birch, A. (2016). **Neural Machine Translation of Rare Words with Subword Units.** ACL. https://arxiv.org/abs/1508.07909. Subword BPE. The executable tutorial uses fixed bytes, with a separate hand-specified BPE example.
4. Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). **Layer Normalization.** https://arxiv.org/abs/1607.06450.
5. Hendrycks, D. and Gimpel, K. (2016). **Gaussian Error Linear Units (GELUs).** https://arxiv.org/abs/1606.08415.
6. Loshchilov, I. and Hutter, F. (2019). **Decoupled Weight Decay Regularization.** ICLR. https://arxiv.org/abs/1711.05101.

# Official implementation documentation

- PyTorch Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
- PyTorch Embedding: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- PyTorch LayerNorm: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- PyTorch GELU: https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
- PyTorch CrossEntropyLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch AdamW: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- PyTorch scaled dot-product attention: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html

Checked for this tutorial on October 8, 2026 UTC. Numerical examples are constructed teaching examples. Training measurements come from the included code and original synthetic corpus.
