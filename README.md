# Toucan

<p align="center">
<a href="https://github.com/UBC-NLP/Toucan/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/UBC-NLP/Toucan"></a>
<a href="https://github.com/UBC-NLP/Toucan/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/UBC-NLP/Toucan"></a>

</p>
 
<img src="./toucan_langs_1.png" width="50%" height="50%" align="right">
<div style='text-align: justify;'>
We address a notable gap in Natural Language Processing (NLP) by introducing a collection of resources designed to improve Machine Translation (MT) for low-resource languages, with a specific focus on African languages. First, We introduce two language models (LMs), Cheetah-1.2B and Cheetah-3.7B, with 1.2 billion and 3.7 billion parameters respectively. Next, we finetune the aforementioned models to create Toucan, an Afrocentric machine translation model designed to support 156 African language pairs. To evaluate Toucan, we carefully develop an extensive machine translation benchmark, dubbed AfroLingu-MT, tailored for evaluating machine translation. Toucan significantly outperforms other models, showcasing its remarkable performance on MT for African languages. Finally, we train a new model, spBLEU_1K, to enhance translation evaluation metrics, covering 1K languages, including 614 African languages. This work aims to advance the field of NLP, fostering cross-cultural understanding and knowledge exchange, particularly in regions with limited language resources such as Africa. 

## Table of Contents
- [1 Our Language Models](#1-Our-Language-Models)
  - [1.1 Training Data](#11-training-data)
  - [1.2 Models Architecture](#12-models-architecture)
  - [1.3 Cheetah Models](#13-cheetah-models)
- [2. AfroNLG Benchmark and Evaluation](#2-our-benchmark-AfroNLG)
  - [2.1 Machine Translation](#21-machine-translation)
  - [2.2 Paraphrase](#22-paraphrase)
  - [2.3 Question Answering](#23-question-answering)
  - [2.4 Summarization](#24-summarization)
  - [2.5 Title Generation](#25-title-generation)
  - [2.6 Cloze](#26-cloze)
- [3. How to use Cheetah model](#3-how-to-use-cheetah-model)
- [4. Ethics](#4-ethics)
- [5. Support Languages](#5-supported-languages)
- [6. Citation](#6-citation)
- [7. Acknowledgments](#7-acknowledgments)

## 1. Our Language Models
## 1.1 Training Data

**Cheetah Training Data**: We are guided by three main principles in developing this data: quality, linguistic diversity, and coverage.

Our collection comprises data from a total of $43$ datasets, encompassing $84$ unique language pairs derived from $46$ different languages. We also develop a new manually translated dataset useful for evaluation in the government domain. In all, the data cover $43$ African languages from five language families domiciled in $29$ African countries. We also include Arabic, English, and French, since these are widely spoken in Africa. Table~\ref{tab:resources} in the Appendix provides detailed information about our collected data, including the number of pairs and data-points (i.e, examples) for each dataset. Table~\ref{tab:data_summary} (Appendix~\ref{appsec:benchmark}) on the other hand has details about each of the $46$ languages in our dataset. These tables serve as a valuable resource for understanding the breadth and depth of our datasets, ensuring transparency and facilitating further research in the field of MT for African languages. We also translate into Yoruba a portion of the Arab-acquis data \cite{habash-etal-2017-parallel} and include it in the benchmark. We refer to this data henceforth as Legal-genre. 
  
## 1.2 Model Architecture

We pretrain Cheetah using the encoder-decoder architecture [(xue-etal-2021-mt5)](https://aclanthology.org/2021.naacl-main.41/). Each of the encoder and decoder components is similar in size and configuration to T5, with 12 layers each with 12 attention heads, and 768 hidden units for the base model. In total, this results in a model with ~580 million parameters. 
## 1.3.  Cheetah Model
For pretraining Cheetah, we use a learning rate of 0.01, a batch size of 1,024 sequences, and a maximum sequence length of 1,024. We pretrain each model for 1M steps. We train our models on Google Cloud TPU with 128 cores (v3-128) from TensorFlow Research Cloud (TFRC).
Cheetah Pytorch and Tenserflow checkpoints are available on Huggingface website for direct download and use ```exclusively for research```. `For commercial use, please contact the authors via email @ (*muhammad.mageed[at]ubc[dot]ca*).`

| **Model**   | **Link** | 
|---------|:------------------:|    
| 🔥**Cheetah-base**🔥: MT5-base model|     [https://huggingface.co/UBC-NLP/cheetah-base](https://huggingface.co/UBC-NLP/cheetah-base)       | 

## 2. AfroNLG Benchmark and Evaluation
We create AfroNLG, a multi-lingual, multi-task benchmark comprising $67$ test sets across six task clusters. Specifically, AfroNLG includes the following: code-swtiching, cloze tasks, machine translation, paraphrase, question answering, summarization, and title generation. AfroNLG supports 517 African languages and language varieties. To the best of our knowledge, this is the most extensive benchmark till date for African languages. 
AfroNLG includes the following tasks: ```machine translation```,  ```paraphrase```,  ```question answering```, ```summarization```, ```title generation```,  ```cloze```.

### 2.1 
#### 2.1  Machine Translation
| **Lang-Pairs**  |**Metric**   |   **mT0** | **mT5** | **Afri-MT5** | **AfriTeVa** |  **Cheetah** |
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:----------:|
| English $\rightarrow$ Afrikaans | Bleu | **20.38<sup>±0.3</sup>**| 12.35<sup>±1.1</sup>|7.12<sup>±2.67 </sup> |  7.75<sup>±1.67</sup> |  19.72<sup>±0.75</sup> |
| English $\rightarrow$ Bemba | Bleu | 19.19<sup>±0.3</sup> | 12.28<sup>±0.48 </sup> |  11.73<sup>±12.3 </sup>|**20.5<sup>±0.87</sup>**    | 18.9<sup>±1.22 </sup> |     
| English $\rightarrow$ Lingala  | Bleu | **15.98<sup>±1.16</sup>** |  14.12<sup>±0.56</sup> |  14.32<sup>±12.74</sup> |  13.88<sup>±1.04</sup> | 9.64<sup>±1.11</sup> |  
| English $\rightarrow$ Rundi    | Bleu | **12.26<sup>±0.47</sup>** |  8.82<sup>±0.43</sup>|9.57<sup>±0.42</sup> |  7.83<sup>±1.04</sup> | 10.54<sup>±0.54</sup> |  
| English $\rightarrow$ Sesotho  | Bleu  | 11.04<sup>±1.2</sup> | 12.74<sup>±0.75</sup> |  10.0<sup>±1.79</sup> |  10.76<sup>±1.4</sup> | **13.3<sup>±1.38</sup>** |  
| English $\rightarrow$ Swahili  | Bleu  | 10.59<sup>±1.84</sup> | 9.33<sup>±0.58</sup>|3.08<sup>±0.57</sup> |  7.24<sup>±0.46</sup> | **11.08<sup>±0.61</sup>** | 
| English $\rightarrow$ Xhosa                    | Bleu            | 10.04<sup>±0.98</sup> | 8.25<sup>±0.7</sup> |  3.86<sup>±1.35</sup> |  7.5<sup>±0.32</sup> |  **12.34<sup>±0.51</sup>** | 
| English $\rightarrow$ Zulu                     | Bleu            | 17.65<sup>±1.86 </sup> | 17.97<sup>±1.69 </sup> |  1.9<sup>±1.11 }    | 13.45<sup>±1.81 </sup> | 19.49<sup>±1.16</sup> |
| English $\rightarrow$ Hausa                    | Bleu            | 5.06<sup>±0.21</sup> | 4.96<sup>±0.16</sup> | 0.85<sup>±0.04</sup> |  7.32<sup>±0.00</sup>   | **9.22<sup>±0.08</sup>** |  
| English $\rightarrow$ Igbo                     | Bleu            | 13.05<sup>±0.17</sup> | 11.57<sup>±0.23</sup> |  1.12<sup>±0.09</sup> |  12.34<sup>±0.23</sup> | **16.75<sup>±0.26</sup>** |
| English $\rightarrow$ Luganda                  | Bleu            | 2.17<sup>±2.77</sup> | 3.33<sup>±0.35</sup> | 0.09<sup>±0.01</sup> |  4.21<sup>±0.77</sup>  | **9.75<sup>±0.01</sup>** |  
| English $\rightarrow$ N. Pidgin                | Bleu            | **33.17<sup>±0.28</sup>** |  32.65<sup>±0.19</sup> |  2.39<sup>±0.23</sup> |  9.39<sup>±0.18</sup> |  32.64<sup>±0.14</sup> |          
| English $\rightarrow$ Swahili                  | Bleu            | 22.04<sup>±2.89</sup> | 23.2<sup>±0.23</sup> |2.79<sup>±0.08</sup> |  22.39<sup>±0.28</sup>  | **28.11<sup>±0.14</sup>** |
| English $\rightarrow$ Zulu                     | Bleu            | 6.83<sup>±0.29</sup>   | 0.58<sup>±1.37 </sup> | 0.4<sup>±0.03</sup>    | 4.45<sup>±0.37</sup> | **11.75<sup>±0.38</sup>** |
| English $\rightarrow$ Twi                      | Bleu            | 3.4<sup>±0.12</sup>  | 1.23<sup>±0.03</sup> |  0.03<sup>±0.0</sup>   | 1.68<sup>±0.94</sup>  | **4.64<sup>±0.13</sup>** | 
| English $\rightarrow$ Yoruba                   | Bleu            | 5.42<sup>±0.85</sup> | 2.58<sup>±3.1</sup> |  0.04<sup>±0.0</sup>     | 3.63<sup>±4.01</sup> | **7.83<sup>±0.14</sup>** |
| English $\rightarrow$ Zulu                     | Bleu            | 10.28<sup>±0.49</sup> | 1.31<sup>±2.26</sup> | 0.14<sup>±0.03</sup> |  3.8<sup>±4.2</sup>  | **12.13<sup>±0.1</sup>** |  
| French $\rightarrow$ Bambara                   | Bleu            | 2.0<sup>±2.6</sup>  | 0.37<sup>±0.19 </sup> | 0.15<sup>±0.01</sup> |  **3.18<sup>±0.18</sup>**    | 3.06<sup>±0.27</sup> |           
| French $\rightarrow$ Ghomálá’                  | Bleu            | 0.4<sup>±0.09</sup> | 0.33<sup>±0.01 </sup> | 0.07<sup>±0.0</sup>     | **0.96<sup>±0.01</sup>**   | 0.28<sup>±0.25</sup> |          
| French $\rightarrow$ Ewe                       | Bleu            | 0.7<sup>±0.35</sup> | 0.31<sup>±0.36</sup> | 0.09<sup>±0.07</sup> |  0.84<sup>±0.16</sup>  | **3.47<sup>±0.03</sup>** |
| French $\rightarrow$ Fon                       | Bleu            | 0.69<sup>±0.31</sup> | 0.8<sup>±0.13 </sup> |  1.52<sup>±0.06 </sup> |  **1.73<sup>±0.53</sup>**    | 1.29<sup>±0.16</sup> |           
| French $\rightarrow$ Moore                     | Bleu            | 0.27<sup>±0.06</sup> | 0.12<sup>±0.05 </sup> | 0.19<sup>±0.02</sup>    | 0.47<sup>±0.04</sup>  | **1.66<sup>±0.86</sup>** |  
| French $\rightarrow$ Wolof                     | Bleu            | **4.02<sup>±0.12</sup>** | 0.3<sup>±0.05 </sup> |  0.11<sup>±0.01</sup>    | 3.08<sup>±0.25</sup>    | 3.01<sup>±0.07</sup> |          
| English $\rightarrow$ N. Pidgin (UNMT)         | Bleu            | **27.44<sup>±0.26</sup>** |  23.42<sup>±1.61</sup> |  7.05<sup>±1.37</sup> |  22.54<sup>±0.84</sup>  | 26.56<sup>±0.04</sup> |         
| Acholi $\rightarrow$ English                   | Bleu            | 16.41<sup>±0.08</sup> | 11.16<sup>±4.77</sup> |  4.9<sup>±0.11</sup>    | 8.37<sup>±8.12</sup>   | **19.33<sup>±0.1</sup>** |  
| Acholi $\rightarrow$ Lugbara                   | Bleu            | 2.57<sup>±0.21</sup> | 1.48<sup>±1.31</sup> | 2.44<sup>±0.37</sup> |  **8.29<sup>±0.14</sup>**    | 7.21<sup>±0.69</sup> |           
| Acholi $\rightarrow$ Luganda                   | Bleu            | 3.64<sup>±0.07</sup> | 1.74<sup>±0.12</sup> | 0.92<sup>±0.01</sup> |  5.53<sup>±0.34</sup>    | **8.03<sup>±0.38</sup>** | 
| Acholi $\rightarrow$ Nyankore                  | Bleu            | 2.17<sup>±0.14</sup> | 0.79<sup>±0.51</sup> | 0.46<sup>±0.03</sup> |  4.26<sup>±0.54</sup>    | **5.1<sup>±0.14</sup>** | 
| Acholi $\rightarrow$ Ateso                     | Bleu            | 1.64<sup>±2.34</sup> | 1.94<sup>±0.25</sup> | 4.9<sup>±0.11</sup>    | **7.74<sup>±0.33</sup>** | 6.33<sup>±0.6</sup> |           
| English $\rightarrow$ Lugbara                  | Bleu            | 6.19<sup>±6.33</sup> | 8.38<sup>±0.49</sup> | 5.93<sup>±0.22</sup> |  10.95<sup>±0.32</sup>    | **11.61<sup>±0.28</sup>** |
| English $\rightarrow$ Luganda                  | Bleu            | 12.08<sup>±0.03</sup> | 10.58<sup>±0.25</sup> |  2.59<sup>±0.73</sup> |  12.41<sup>±0.35</sup> | **17.12<sup>±0.16</sup>** |
| English $\rightarrow$ Nyankore                 | Bleu            | 6.46<sup>±0.08</sup> | 5.69<sup>±0.02</sup> | 1.4<sup>±0.39</sup> | 7.88<sup>±0.18</sup> | **9.04<sup>±0.24</sup>** |
| English $\rightarrow$ Ateso (salt)             | Bleu            | 10.24<sup>±0.06</sup> | 8.28<sup>±0.19</sup> | 4.91<sup>±0.59</sup> |  **11.64<sup>±0.49</sup>** |  11.12<sup>±0.38</sup> |         
| Lugbara $\rightarrow$ Ateso                    | Bleu            | 2.21<sup>±0.35</sup> | 1.5<sup>±0.2</sup>   | 2.22<sup>±0.15</sup> |  **6.67<sup>±0.32</sup>**    | 3.68<sup>±0.31</sup> |         
| Luganda $\rightarrow$ Lugbara                  | Bleu            | 3.96<sup>±0.57</sup> | 2.61<sup>±0.12</sup> | 3.44<sup>±0.32</sup> |  **8.05<sup>±0.23</sup>**    | 7.99<sup>±0.47</sup> |        
| Luganda $\rightarrow$ Ateso                    | Bleu            | 4.47<sup>±0.08</sup> | 3.01<sup>±0.16</sup> | 2.5<sup>±0.22</sup>    | **8.17<sup>±0.18</sup>**    | 8.13<sup>±0.33 </sup> |        
| Nyankore $\rightarrow$ Lugbara                 | Bleu            | 3.45<sup>±0.29</sup> | 2.1<sup>±0.32</sup>  | 2.6<sup>±0.29</sup>    | **7.5<sup>±0.09</sup>**     | 7.29<sup>±0.09</sup> |          
| Nyankore $\rightarrow$ Luganda                 | Bleu            | 8.54<sup>±0.17</sup> | 6.91<sup>±0.23</sup> |2.01<sup>±0.25 </sup> |  **6.77<sup>±6.73</sup>**    | 6.25<sup>±10.26</sup> |         
| Nyankore $\rightarrow$ Ateso                   | Bleu            | 3.33<sup>±0.11</sup> | 2.25<sup>±0.23</sup> |2.12<sup>±0.4</sup>    | 6.27<sup>±0.12</sup>     | **6.36<sup>±0.4</sup>** |

#### 2.2  Paraphrase
| **Langs**  |**Metric**   |   **mT0** | **mT5** | **Afri-MT5** | **AfriTeVa** |  **Cheetah** |
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:----------:|                                        
| Multilingual   | Bleu  | 41.79<sup>±0.28</sup> | 41.75<sup>±0.21 </sup> | 34.72<sup>±0.51</sup> | 43.02<sup>±1.25</sup> | **43.23<sup>±0.09</sup>** | 
| Berber   | Bleu | 44.84<sup>±0.31</sup> | 44.03<sup>±0.24</sup> | 36.08<sup>±0.83</sup>   | **46.41<sup>±0.71</sup>** |46.0<sup>±0.27</sup> |          
| Kabyle   | Bleu | 25.91<sup>±0.13</sup> | 25.32<sup>±0.46</sup> | 11.56<sup>±0.73</sup>   | 16.06<sup>±14.79</sup> | **26.27<sup>±0.56</sup>** |


#### 2.3  Question Answering

| **Langs**  |**Metric**   |   **mT0** | **mT5** | **Afri-MT5** | **AfriTeVa** |  **Cheetah** |
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:----------:|
| QA Swahili   | F1         | **79.84<sup>±0.19</sup>** | 72.04<sup>±0.54</sup> | 0      | 62.64<sup>±0.78</sup>   | 71.98<sup>±1.18</sup>|         


#### 2.4  Summarization

| **Langs**  |**Metric**   |   **mT0** | **mT5** | **Afri-MT5** | **AfriTeVa** |  **Cheetah** |
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:----------:|
| Multilingual  | RougeL | 22.31<sup>±0.12</sup> | 22.23<sup>±0.04</sup>| 5.34<sup>±0.48</sup>  | 18.97<sup>±0.06</sup> | 24.86<sup>±0.02 </sup>|                                                             | Amharic  | RougeL | 13.81<sup>±0.04</sup> | 13.09<sup>±0.03</sup> | 4.4<sup>±1.07</sup> | 8.29<sup>±0.51</sup>  | **15.09<sup>±0.1</sup>** |                                                  
| Igbo   | RougeL | **18.9<sup>±0.73</sup>** | 13.22<sup>±0.46</sup> | 14.24<sup>±0.39</sup> | 16.05<sup>±0.49</sup> | 17.36<sup>±0.43</sup> |                                                                
| Oromo  | RougeL | 11.28<sup>±0.03</sup> | 10.51<sup>±0.07</sup> | 3.52<sup>±0.49</sup> | 7<sup>±1.73</sup>  | **14.53<sup>±0.1</sup>** |
| Rundi  | RougeL | 19.63<sup>±0.01 </sup> | 18.02<sup>±0.13 </sup> |11.82<sup>±0.39 </sup> | 16.13<sup>±0.03</sup> | **22.57<sup>±0.04</sup>** | 
| Swahili | RougeL | 26.38<sup>±0.02</sup> | 24.81<sup>±0.11</sup> |15.07<sup>±0.17</sup> | 21.59<sup>±0.13</sup> | **29.05<sup>±0.13</sup>** | 
| Yoruba  | RougeL | 21.57<sup>±0.05</sup> | 20.06<sup>±0.12</sup> |13.52<sup>±0.18</sup> | 17.3<sup>±0.11</sup>  | **22.49<sup>±0.0</sup>** | 
| Hausa   | RougeL | 26.46<sup>±0.06</sup> | 25.76<sup>±0.02</sup> |19.96<sup>±0.26</sup> | 25.19<sup>±0.11</sup> | **30.07<sup>±0.31</sup>** | 
| Nigerian Pidgin | RougeL | 26.54<sup>±0.05</sup> | 25.79<sup>±0.1</sup> | 14.28<sup>±1.23</sup> | 20.29<sup>±0.12</sup> | **27.08<sup>±0.02</sup>** | 
| Somali | RougeL | 20.69<sup>±0.08</sup> | 19.21<sup>±0.06</sup> |13.62<sup>±0.81</sup> | 19.27<sup>±0.18</sup> | **23.92<sup>±0.04</sup>** |
| Tigrinya | RougeL | 15.84<sup>±0.13</sup> | 13.93<sup>±0.11</sup> | 6.53<sup>±0.42</sup>   | 10.07<sup>±0.09</sup> | **16.88<sup>±0.12</sup>** |

     
#### 2.5  Title Generation

| **Langs**  |**Metric**   |   **mT0** | **mT5** | **Afri-MT5** | **AfriTeVa** |  **Cheetah** |
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:----------:|
| Multilingual   | Bleu | 6.53<sup>±0.02</sup> | 6.65<sup>±0.08</sup> | 0.1<sup>±0.02</sup>    | 5.2<sup>±0.02</sup>  | **7.52<sup>±0.07</sup>**  |  
| Amharic  | Bleu | 3.13<sup>±0.23</sup>   | 2.65<sup>±0.68</sup> | 0.34<sup>±0.14</sup>   | 2.31<sup>±0.14</sup> | **4.34<sup>±0.34</sup>**  |  
| Igbo  | Bleu | 6.95<sup>±0.13</sup>  | 6.9<sup>±0.22</sup>  | 0.77<sup>±0.12</sup>   | 4.61<sup>±0.14</sup>  | **8.47<sup>±0.07</sup>**  |  
| Oromo | Bleu | 1.1<sup>±1.84</sup> | 2.66<sup>±0.19</sup> | 0.21<sup>±0.06</sup>   | 1.54<sup>±0.17</sup>    | **3.26<sup>±0.21</sup>**  |  
| Rundi | Bleu | 4.4<sup>±0.28</sup> | 4.13<sup>±0.22</sup> | 0.84<sup>±0.07</sup>   | 3.33<sup>±0.23</sup>    | **6.05<sup>±0.5</sup>**  |   
| Swahili | Bleu | 9.1<sup>±0.23</sup> | 9.31<sup>±0.11</sup> | 1.22<sup>±0.09</sup>   | 7.01<sup>±0.09</sup>  | **10.59<sup>±0.6</sup>**  |  
| Yoruba | Bleu  | 6.8<sup>±0.16</sup> | 7.23<sup>±0.59</sup> | 0.34<sup>±0.05</sup>    | 5.04<sup>±2.0</sup>  | **7.97<sup>±0.32</sup>**  |  
| Hausa  | Bleu  | 8.11<sup>±0.24 </sup>  | 7.3<sup>±0.34</sup>   | 2.59<sup>±0.01</sup>   | 6.69<sup>±0.18</sup> | **8.48<sup>±0.23</sup>**  |   
| Nigerian Pidgin | Bleu  | **6.75<sup>±0.6</sup>** | 3.96<sup>±4.3</sup>   | 0.89<sup>±0.02</sup>   | 4.72<sup>±0.84</sup> | 6.22<sup>±0.28</sup>  |           
| Somali  | Bleu  | 3.37<sup>±0.21</sup> | 3.31<sup>±0.16</sup> | 0.38<sup>±0.11</sup>   | 2.82<sup>±0.47</sup>  | **5.25<sup>±0.14</sup>**  |  
| Tigrinya  | Bleu | 2.99<sup>±0.1</sup> | 2.94<sup>±1.09</sup> | 0.7<sup>±0.18</sup>    | 1.92<sup>±0.26</sup>  | **5.1<sup>±0.05</sup>**  |   


#### 2.6  Cloze

| **Task**  |**Metric**   |   **mT0** | **mT5** | **Afri-MT5** | **AfriTeVa** |  **Cheetah** |
|----------------|:---------:|:-------------:|:-----------:|:----------:|:----------:|:----------:|
| Mask-one - 517 Languages | Bleu  | 13.61<sup>±0.91</sup> | 8.18<sup>±3.94</sup> | 0.00<sup>±0.00</sup>   | 8.36<sup>±3.42</sup>  | **13.98<sup>±0.32</sup>** |
| Mask-at-least-one - 517 Languages | Bleu | 2.36<sup>±0.11</sup> | 2.66<sup>±0.09</sup> | 0.93<sup>±0.12</sup>  | 0.68<sup>±0.09</sup> | **7.07<sup>±0.09</sup>** |




#  3. How to use Cheetah model

Below is an example for using **Cheetah** predict masked tokens. 
``` bash
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

tokenizer = T5Tokenizer.from_pretrained("UBC-NLP/cheetah-base")
model = AutoModelForSeq2SeqLM.from_pretrained("UBC-NLP/cheetah-base")

yor_prompt="ìròyìn kan nípa owó ìjọba <extra_id_0> kan"

input_ids = tokenizer(yor_prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print("Tokenized input:", tokenizer.tokenize(yor_prompt))
print("Decoded output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

```
Output:
```bash
Tokenized input: ['▁ìròyìn', '▁kan', '▁nípa', '▁owó', '▁ìjọba', '<extra_id_0>', '▁kan']
Decoded output:  ìpínlẹ̀
```

## 4. Ethics
Cheetah aligns with Afrocentric NLP where the needs of African people is put into consideration when developing technology. We believe Cheetah will not only be useful to speakers of the languages supported, but also researchers of African languages such as anthropologists and linguists.
We discuss below some use cases for Cheetah and offer a number of broad impacts. 
- Cheetah aims to address the lack of access to technology in about 90% of the world's languages, which automatically discriminates against native speakers of those languages. More precisely, it does so by focusing on Africa.
  To the best of our knowledge, Cheetah is the first massively multilingual PLM developed for African languages and language varieties. A model with knowledge of <b>517 African languages</b>, is by far the largest to date for African NLP. 
- Cheetah enables improved access of important information to the African community in Indigenous African languages. This is especially beneficial for people who may not be fluent in other languages. This will potentially connect more people globally. 
- Cheetah affords opportunities for language preservation for many African languages. To the best of our knowledge, Cheetah consists of languages that have not been used for any NLP task until now.
  We believe that it can help encourage  continued use of these languages in several domains, as well as trigger future development of language technologies for many of these languages.
- Cheetah Although LMs are useful for a wide range of applications, they can also be misused. Cheetah is developed using publicly available datasets that may carry biases.
  Although we strive to perform analyses and diagnostic case studies to probe performance of our models, our investigations are by no means comprehensive nor guarantee absence of bias in the data.
  In particular, we do not have access to native speakers of most of the languages covered. This hinders our ability to investigate samples from each (or at least the majority) of the languages.
     
## Supported languages
Please refer to [**supported-languages**](supported-languages.txt)

## Citation
If you use the pre-trained model (Cheetah) for your scientific publication, or if you find the resources in this repository useful, please cite our paper as follows (to be updated):

```
@misc{adebara2024cheetah,
      title={Cheetah: Natural Language Generation for 517 African Languages}, 
      author={Ife Adebara and AbdelRahim Elmadany and Muhammad Abdul-Mageed},
      year={2024},
      eprint={2401.01053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgments
We gratefully acknowledges support from Canada Research Chairs (CRC), the Natural Sciences and Engineering Research Council of Canada (NSERC; RGPIN-2018-04267), the Social Sciences and Humanities Research Council of Canada (SSHRC; 435-2018-0576; 895-2020-1004; 895-2021-1008), Canadian Foundation for Innovation (CFI; 37771), [Digital Research Alliance of Canada](https://alliancecan.ca), [UBC ARC-Sockeye](https://arc.ubc.ca/ubc-arc-sockeye), Advanced Micro Devices, Inc. (AMD), and Google. Any opinions, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of CRC, NSERC, SSHRC, CFI, the Alliance, AMD, Google, or UBC ARC-Sockeye.
