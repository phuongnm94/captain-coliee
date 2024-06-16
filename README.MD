# CAPTAIN at COLIEE 2024: Large Language Model for Legal Text Retrieval and Entailment

Recently, the Large Language Models (LLMs) have made a great contribution to massive Natural Language Processing  (NLP) tasks. 
This year, our team, CAPTAIN, utilizes the power of LLM for legal information extraction tasks of the COLIEE competition. To this end, the LLMs are used to understand the complex meaning of legal documents, summarize the important points of legal statute law articles as well as legal document cases, and find the relations between them and specific legal cases. By using various prompting techniques, we explore the hidden relation between the legal query case and its relevant statute law as supplementary information for testing cases.  The experimental results show the promise of our approach, with first place in the task of legal statute law entailment, competitive performance to the State-of-the-Art (SOTA) methods on tasks of legal statute law retrieval, and legal case entailment in the COLIEE 2024 competition.  
##  Results: 
1. **Task 1 : The Legal Case Retrieval task**
                    
    | Run                                                                                      | Dev    | Private Test (2024) |     |
    | ---------------------------------------------------------------------------------------- | ------ | ------------------- | --- |
    | [TQM]                                                                                    |        | 0.4432              |     |
    | [UMNLP]                                                                                  |        | 0.4134              |     |
    | [YR]                                                                                     |        | 0.3605              |     |
    | [JNLP]                                                                                   |        | 0.3246              |     |
    |                                                                                          |        |                     |     |
    | *(This work)*                                                                            |        |                     |     |
    | **captainFT5** [[Ours](https://link.springer.com/chapter/10.1007/978-981-97-3076-6_9)]   | 0.3534 | 0.1688              |     |
    | **captainMSTR**  [[Ours](https://link.springer.com/chapter/10.1007/978-981-97-3076-6_9)] | 0.3681 | 0.1574              |     |
    | ...                                                                                      |        |                     |     |
    |                                                                                          |        |                     |     |
   
2.  **Task 2 (Second place): The Legal Case Entailment task**

    This task involves the identification of a paragraph from existing cases that entails the decision of a new case.

    Given a decision Q of a new case and a relevant case R, a specific paragraph that entails the decision Q needs to be identified. We confirmed that the answer paragraph can not be identified merely by information retrieval techniques using some examples. Because the case R is a relevant case to Q, many paragraphs in R can be relevant to Q regardless of entailment.

    This task requires one to identify a paragraph which entails the decision of Q, so a specific entailment method is required which compares the meaning of each paragraph in R and Q in this task. 
    
    | Run                                                                                     | Dev       | Private Test (2024) |     |
    | --------------------------------------------------------------------------------------- | --------- | ------------------- | --- |
    | [[CAPTAIN 2023](https://arxiv.org/abs/2401.03551)]                                      | 75.45     | _                   |     |
    | [AMHR]                                                                                  |           | 65.12               |     |
    | [JNLP]                                                                                  |           | 63.20               |     |
    | [NOWJ]                                                                                  |           | 61.17               |     |
    |                                                                                         |           |                     |     |
    | **captainZs2** [[Ours](https://link.springer.com/chapter/10.1007/978-981-97-3076-6_9)]  | **76.36** | 63.35               |     |
    | **captainZs3**  [[Ours](https://link.springer.com/chapter/10.1007/978-981-97-3076-6_9)] | 74.55     | 62.35               |     |
    | **captainFs2** [[Ours](https://link.springer.com/chapter/10.1007/978-981-97-3076-6_9)]  | 70.13     | 63.60               |     |
    | ...                                                                                     |           |                     |     |
    |                                                                                         |           |                     |     |
3.  **Task 3 (First place): The Statute Law Retrieval Task**

    The COLIEE statute law competition focuses on two aspects of legal information processing related to answering yes/no questions from Japanese legal bar exams (the relevant data sets have been translated from Japanese to English).

    Task 3 of the legal question answering task involves reading a legal bar exam question Q, and extracting a subset of Japanese Civil Code Articles S1, S2,..., Sn from the entire Civil Code which are those appropriate for answering the question such that

    Entails(S1, S2, ..., Sn , Q) or Entails(S1, S2, ..., Sn , not Q).

    Given a question Q and the entire Civil Code Articles, we have to retrieve the set of "S1, S2, ..., Sn" as the answer of this track. 

    ---
    
    Results of Task 3 on the development set (using F2 score), the underlined settings play as the main sub-method in the ensemble various methods. The notations (*)  refer to the runs using LLM with undisclosed training data,  (#)  refer to the runs do not use LLM.  

    | Run                                                                      | R02       | R03       | R04       | R05 (private test) |
    | ------------------------------------------------------------------------ | --------- | --------- | --------- | ------------------ |
    | JNLP.constr-join (*)                                                     |           |           |           | **74.08**          |
    | TQM-run1 (#)                                                          |           |           |           | 71.71              |
    | NOWJ-25mulreftask-ensemble (#)                                       |           |           |           | 71.23              |
    | JNLP.Mistral(*)                                                          |           |           |           | **74.08**          |
    | DatFlt-q+MonoT5 (#) [[CAPTAIN 2023](https://arxiv.org/abs/2401.03551)] | **76.36** | 85.52     | 75.69     | _                  |
    |                                                                          |           |           |           |                    |
    | *(This work)*                                                            |           |           |           |                    |
    | Mckpt (#)                                                              | 69.49     | 78.35     | 77.54     | 71.35              |
    | MonoT5                                                                   | 71.54     | 79.83     | 58.96     | 55.56              |
    | Mckpt --> LLM-re-ranking                                                 | 66.67     | 78.87     | 75.44     | 70.64              |
    |                                                                          |           |           |           |                    |
    | LLM-re-ranking + MonoT5                                                  | 73.87     | 84.04     | 77.87     | 72.68              |
    | Mckpt + MonoT5                                                           | 76.13     | 85.17     | 78.23     | 71.71              |
    | Mckpt + MonoT5 + LLM-re-ranking                                          | 75.04     | **85.68** | **79.94** | 73.35              |
    |                                                                          |           |           |           |                    |

4.  **Task 4 (First place): The Statute Law Entailment task** 
   
    In this task, competitors need to answer the given queries by using the information that lies inside the relevant articles which is given along with the queries. To find correct answers, the conditions inside the queries should be matched with the conditions described in the legal articles. The statement of the relevant articles will be used to determine the answer to the query.
    | Run                                          | R04    | R02    | R01    | R05 (private test) |
    | -------------------------------------------- | ------ | ------ | ------ | ------------------ |
    | JNLP1                                        |        |        |        | 0.8165             |
    | UA_slack                                     |        |        |        | 0.7982             |
    | UA_gpt                                       |        |        |        | 0.7798             |
    | AMHR.ensembleA50                             |        |        |        | 0.7706             |
    | NOWJ.pandap46                                |        |        |        | 0.7523             |
    | HI1                                          |        |        |        | 0.7523             |
    | OVGU1                                        |        |        |        | 0.7064             |
    |                                              |        |        |        |                    |
    | *(This work)*                                |        |        |        |                    |
    | Auto-CoT  (CAPTAIN3)                         | 0.8415 | 0.8395 | 0.7207 | 0.7890             |
    | Fine-tuning Few-shot   (CAPTAIN1)            | 0.8217 | 0.8148 | 0.7748 | 0.7890             |
    | Fine-tuning Data Augmentation     (CAPTAIN2) | 0.8415 | 0.7901 | 0.7568 | **0.8257**         |
    |                                              |        |        |        |                    |
## Runs: 
Check our runs task 1, task 2, task 3, task 4 in corresponding folders, respectively:  `exp_t1`, `exp_t2_2023`, 
`exp_t3`, `exp_t4`, 
## Citation
 
```bib
@InProceedings{10.1007/978-981-97-3076-6_9,
author="Nguyen, Phuong
and Nguyen, Cong
and Nguyen, Hiep
and Nguyen, Minh
and Trieu, An
and Nguyen, Dat
and Nguyen, Le-Minh",
editor="Suzumura, Toyotaro
and Bono, Mayumi",
title="CAPTAIN at COLIEE 2024: Large Language Model for Legal Text Retrieval and Entailment",
booktitle="New Frontiers in Artificial Intelligence",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="125--139",
abstract="Recently, the Large Language Models (LLMs) has made a great contribution to massive Natural Language Processing (NLP) tasks. This year, our team, CAPTAIN, utilizes the power of LLM for legal information extraction tasks of the COLIEE competition. To this end, the LLMs are used to understand the complex meaning of legal documents, summarize the important points of legal statute law articles as well as legal document cases, and find the relations between them and specific legal cases. By using various prompting techniques, we explore the hidden relation between the legal query case and its relevant statute law as supplementary information for testing cases. The experimental results show the promise of our approach, with first place in the task of legal statute law entailment, competitive performance to the State-of-the-Art (SOTA) methods on tasks of legal statute law retrieval, and legal case entailment in the COLIEE 2024 competition. Our source code and experiments are available at https://github.com/phuongnm94/captain-coliee/tree/coliee2024.",
isbn="978-981-97-3076-6"
}
```
[[Online version](https://link.springer.com/chapter/10.1007/978-981-97-3076-6_9) ]  
##  License
MIT-licensed. 




 
