# Attribute Alignment: Controlling Text Generation from Pre-trained

Code for the paper:

[**Attribute Alignment: Controlling Text Generation from Pre-trained**](https://aclanthology.org/2021.findings-emnlp.194.pdf) by [**Dian Yu**](), [**Zhou Yu**]() and [**Kenji Sagae**](). EMNLP 2021 Findings.


## Abstract
Large language models benefit from training with a large amount of unlabeled text, which gives them increasingly fluent and diverse generation capabilities. However, using these models for text generation that takes into account target attributes, such as sentiment polarity or specific topics, remains a challenge. We propose a simple and flexible method for controlling text generation by aligning disentangled attribute representations. In contrast to recent efforts on training a discriminator to perturb the token level distribution for an attribute, we use the same data to learn an alignment function to guide the pre-trained, non-controlled language model to generate texts with the target attribute without changing the original language model parameters. We evaluate our method on sentiment- and topic-controlled generation, and show large performance gains over previous methods while retaining fluency and diversity.

## Bibtex:
<pre>
@inproceedings{yu-etal-2021-attribute-alignment,
    title = "Attribute Alignment: Controlling Text Generation from Pre-trained Language Models",
    author = "Yu, Dian  and
      Yu, Zhou  and
      Sagae, Kenji",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.194",
    doi = "10.18653/v1/2021.findings-emnlp.194",
    pages = "2251--2268",
    abstract = "Large language models benefit from training with a large amount of unlabeled text, which gives them increasingly fluent and diverse generation capabilities. However, using these models for text generation that takes into account target attributes, such as sentiment polarity or specific topics, remains a challenge. We propose a simple and flexible method for controlling text generation by aligning disentangled attribute representations. In contrast to recent efforts on training a discriminator to perturb the token level distribution for an attribute, we use the same data to learn an alignment function to guide the pre-trained, non-controlled language model to generate texts with the target attribute without changing the original language model parameters. We evaluate our method on sentiment- and topic-controlled generation, and show large performance gains over previous methods while retaining fluency and diversity.",
}
}
</pre>
