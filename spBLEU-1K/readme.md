# spBLEU<sup>1K</sup>
Employing the BLEU metric for evaluating translations, particularly in the context of low-resource languages, is suboptimal due to its fundamental reliance on n-gram overlap. This reliance significantly impacts the metric's effectiveness, as it is heavily influenced by the specific tokenization method used. Notably, employing a more aggressive tokenization strategy can lead to artificially inflated BLEU scores~\newcite{goyal2022flores}. To address this,[Goyal et al. (2022)](https://aclanthology.org/2022.tacl-1.30/) proposed a novel metric, SentencePiece BLEU (i.e., ``spBLEU``}), designed to measure and analyze the performance of translations across 101 languages. This approach involves training a new SentencePiece-based tokenizer using monolingual data for 101 languages, replacing the default tokenizer typically used in SacreBLEU, known as `mosestokenizer' [(Post, 2018)](https://aclanthology.org/W18-6319/). This innovation aims to standardize the tokenization process, thus providing more accurate and comparable translation performance metrics across many languages.

Significantly, the spBLEU metric covers merely 23 out of the 43 languages present in our AfroLingu-MT. To address this limitation, we adopt a methodology similar to that of [Goyal et al. (2022)](https://aclanthology.org/2022.tacl-1.30/). Namely, we develop a new SentencePiece tokenizer (i.e., spBLEU<sup>1K</sup>) that utilizes 1000+ monolingual data sources.

## Training Data
We collect monolingual data covering 1,003 languages, including 614 African languages, 53 Indigenous American languages, and the remainder spanning the most resource-rich languages worldwide. We use a diverse data source, encompassing Wikipedia, Wikibooks, the Bible, newspapers, and common web sources. Additionally, we utilize the MADLAD dataset~\cite{kudugunta2023madlad400}, which covers 419 languages. 


## Training a SentencePiece Model (SPM)
One significant challenge is the uneven availability of monolingual data across various languages. This disparity is especially acute for low-resource languages, which often suffer from a lack of comprehensive coverage in subword units and may not possess a sufficiently large corpus of sentences to ensure a broad and diverse representation of content. To address this issue and enhance the training of our new SPM, we adopt a temperature upsampling technique similar to the methodology described in [Conneau et al. (2019)](https://arxiv.org/abs/1911.02116)

## Integrating with SacreBleu
We integrate this newly created SPM into SacreBLEU, resulting in the formulation of our more inclusive metric spBLEU<sup>1K</sup>. Our metric is thus designed to provide a more comprehensive evaluation across a broader range of languages, including those that are underrepresented in existing metrics such as spBLEU.


## Getting Start
### Install requirments 

- Installing the integrated version of sacreBLEU that supports the spBLEU<sup>1K</sup> tokenizer.

``` bash
    git clone https://github.com/UBC-NLP/Toucan.git
    cd spBLEU-1K/sacrebleu
    pip install -e .
```

- Installing Huggingface's evaluate library.
 ``` bash
    pip install evaluate
```

### Scoring

``` python
import evaluate
metric = evaluate.load("sacrebleu")


predictions = ["ወንድሞች ሆይ፥ በዚህ ዓለም እንግዶችና እንግዶች እንደ መሆናችሁ መጠን፥ ሁልጊዜ እርስ በርሳችሁ ከሚዋጋ ሥጋዊ ምኞት ርሰት እንዳትወድቁ እለምናችኋለሁ።"]
references = [["ወዳጆች ሆይ፥ ነፍስን ከሚዋጋ ሥጋዊ ምኞት ትርቁ ዘንድ እንግዶችና መጻተኞች እንደ መሆናችሁ እለምናችኋለሁ፤"]]


#Score using the default SentencePiece tokenizer (spBLEU).
results = metric.compute(tokenize="spm", predictions=predictions, references=references)
print(round(results["score"], 1))
# 33.6

#Score using the spBLEU-1K SentencePiece tokenizer.
results = metric.compute(tokenize="spBLEU-1K", predictions=predictions, references=references)
print(round(results["score"], 1))
# 41.5

```