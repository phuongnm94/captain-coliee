

--- 
### Meeting on Dec 21, 2023 

> **Important Note**: If we have annotated data (previous years to 2023 + private_test 2024)
=> train = data (previous -> 2021), public_dev = data2022, public_test = data2023
=> all the results are required to optimized on the public_dev (data2022, or previous years if necessary) and test again on the public_test (data2023). => best setting is then used for private_test data2024.

- Task 1 (Minh)
  - previous week: 
    - SOTA 2023: F1 0.3 Thuirs
    - Basline (Minh): chua ra duoc % 
      - sd graph: 
        - motivation: consider to the link between articles and ranking between them. 
        - 1 node la 1 article (directed graph), 
        - 2 node is connected if it is reffered.  
  - next week:
    - [!] sua lai Eval metric, query level thay vi candidate level.  
    - [!] Chay lai SOTA 2023, Thuirs. 
    - Tim candidate nodes: Tfidf, bm25     
    - Sd node embedding khac: sd Roberta / LegalBERT 
- Task 2 (Hiep)
  - previous week:
    - SOTA 2024: 74.77 (our reproduce) 74.56(paper)
    - Baseline:  
      - Bm25 31.xx
      - negative sampling 74.77 (our reproduce) 
      - thong tin parsing hien tai: entail fragment + input paragraph
      - thong tin chua su dung: base-case (co the co huu ich)
  - next week:
    - check thong tin base-case. 
    - check lai negative sampling cua Thanh. 
      - sd 1 phuong phap khac de negative sampling (train 1 cai LegalBERT).
    - [!] enssemble: check cai nam ngoai cua Thanh kq truoc enssemble xem duoc bn.  
    
- Task 3 (Phuong, Dat) 
  - previous week: 
    - SOTA 2023: 
      - dev R02: 76.36
      - public test R03: 85.52 
      - private test R04: 75.69
    - Baseline (Mckpt setting):
      - dev R02: 70.31 
      - public test R03: 76.94
      - public test2 R04: 77.17
  - next week:
    - Rerun mono-T5 for enssembling 
    - features extraction by Llama-2 model. 
  
- Task 4 (Cong, An) 
  - previous week:
    - SOTA 2023: 78.81 (our reproduce) 78.22(paper)
    - Baseline: 
      - T5 xxl, rezo shot learning. 
        - input: prompt (q + tat ca A) => chon prompt tu prompt lib, dua tren dev => pick best prompt de eval tren test. 
  - next week:
    - thu llama-2-chat-hf/llama-2-hf [code_example](./llm_bio_extraction.md)
    - thu voi `mistral` model from `mistral.ai` (llama2)
    - thu few shot: 
      - tim **dynamic** candidates dua tren sBert
    - thu CoT (chain of thought)
      - tao - samples **fix** candidates, trong samples huong dan cach model infer output. 
--- 
### Meeting on Dec 28, 2023 
    
- Task 1 (Minh)
  - previous week:
    - [!] sua lai Eval metric, query level thay vi candidate level.  
    - [!] Chay lai SOTA 2023, Thuirs. 
    - Tim candidate nodes: Tfidf, bm25     
    - Sd node embedding khac: sd Roberta / LegalBERT 
    - 
    - Method:
        - using BM25 to rank candidate:
            - Query = top 20 keyword TF-IDF
            - Corpus = whole document
        - Train GNN for link prediction
            - Node embedding: mean Legal-BERT sentence embeddings
            - layer neighbors (pytorch-geometric)
            
    - Result: 
      - Precision: 0.11526627218934911
      - Recall: 0.5860409145607701
      - F1: 0.19264240506329114
    - Problem:
        - train documents $\cap$  test documents $= \empty$
  - next week:
    - [!] Recall score of Bm25 va Tfidf (top 50, 80, 100, 150)?
    - [!] Su dung MonoT5 fine-tune cap relevant. => Su dung code train cua task 2.
    - BM25:
        - Query = chunk contains <FRAGMENT_SUPPRESSED>
    - GNN: inductive link prediction ?
    - Other method to increase Precision?
        - Finetune LLM to classify relevant
        - CoT LLM to construct the query
- Task 2 (Hiep)
  - Previous week
    - SOTA negative sampling:
      - First, choose top 10 negative samples based on BM25
      - Then use negative and positive samples to train MonoT5
      - After training, resample negative samples again with new fine-tuned model
    - Try negative sampling with other models (ex. LegalBERT): in progress
    - Check relevance between base cases and entailed fragment, positive paragraphs
      - The information in entailed fragment can be found in the positive paragraph
      - Can we find the relevant paragraphs in the base cases and combine with entailed fragment to find the positive paragraph?
  - Next week
    <!-- - Try negative sampling with LegalBert -->
    - step1: Fine-tuned Mono T5 1 query case, all case -> get top-k (k=10) for similar case. 
    - step2: Fine-tuned Mono T5 1 query case, k retrieved case from step 1 -> final model. 
    - Check relevance between base cases and entailed fragments
      - Try Rerank MotoT5 to search for relevant paragraphs in base cases

- Task 3 (Phuong, Dat) 
  - previous week:
    - [!] Rerun mono-T5 for enssembling => fix bug confict 
    - [!] features extraction by Llama-2 model => [x]
    - Re-run settings: Data-filter Articles (DatFltA) and similar Queries (DatFltQ). 
    - Result: 
      | Setting   |      R02 (dev)     |  R03 (test 1) |  R04 (test 2) |
      |----------|-------------:|------:|------:|
      | SOTA 2023 |  76.36| 85.52 | 75.69|
      | -- |      |
      | Baseline (!) (Mckpt) |    69.59   |   77.54 |76.21 |
      | Baseline (Mckpt) |    68.86    |   77.21  | 72.48 |
      | Mckpt + Data-filter Q |    68.80   |   76.18 | 72.48 |
      | Mckpt + Data-filter A |   68.86    |   77.21  | 72.48 |
      | Mckpt + Data-filter A + Q |    68.80   |   76.18 | 72.48 |
      | -- |     |
      | Mono T5 (zero)|    ?   |   ? | ? |
      | Mono T5 (ft)|    ?   |   ? | ? |
      <!-- | Baseline (Mckpt) |    70.31   |   76.94 |77.17 | -->
  - next week:
    - optimize hyper-parameters fine-tuned model Bert Japanese. 
    - [!] Rerun mono-T5 for enssembling  
    - [!] features extraction by Llama-2 model. 
    - summarize doc and re-ranking by LLMs.  => dua toan bo articles  / hoac dua tung cai.
- Task 4 (Cong, An) 
  - previous week:
    - Few-Shot prompting. 
      - Step1: Sử dụng Dense Retrieval để tính các statute tương tự nhau
      - Step2: Sử dụng top 3 statute giống statute query nhất để làm 3 ví dụ cho few-shot
      - **Result** (+2%)
    - Few-Shot + CoT prompting: 
      - Ap dụng thêm CoT cho cả 3 ví dụ trên xong ghép lại thành 3-shot cho statute query
      - **Result**:  TestR03: (76%-> 82%), TestR04: (78% -> 83%)
     ## Cách xây dựng fewshot + CoT:
      ### Step1: Sinh ra chain tự động từ tập train theo mẫu prompt "Let's think step by step why the answer is ..."
      Ví dụ: 
        ```"Article 5
            (1) A minor must obtain the consent of the minor's legal representative to perform a juridical act;provided, however, that this does not apply to a juridical act for merely acquiring a right or being released from               an obligation.
            (2) A juridical act in contravention of the provisions of the preceding paragraph is voidable.
            (3) Notwithstanding the provisions of paragraph (1), a minor may freely dispose of property that the legal representative has permitted the minor to dispose of for a specified purpose, to an extent that falls within             the scope of that purpose. The same applies if the minor disposes of property that the legal representative has permitted the minor to dispose of without specifying a purpose.
            Question: A contract of sales concluded by a minor may not be rescinded if it relates to daily life, even in cases the consent of the parental authority is not obtained. True or False?
            Answer: False
            
            Let't think step by step why the answer is False
            A contract of sales concluded by a minor may be rescinded if it relates to daily life, even if the consent of the parental authority is not obtained. So the final answer is False."
      ```
      ### Step2: Sử dụng Dense Passage Retrival để lựa chọn ra các statute tương tự với query statue.

      ### Step3: Ghép các statute tương đồng lại thành fewShot theo format: "prompting{example1} Let's think step by step \n Answer: {CoT}"
        Ví dụ: 
      ```"Article 5 (1) A minor must obtain the consent of the minor's legal representative to perform a juridical act;provided, however, that this does not apply to a juridical act for merely acquiring a right or being released from an obligation.
          (2) A juridical act in contravention of the provisions of the preceding paragraph is voidable.
          Article 6 (1) A minor who is permitted to conduct one or multiple types of business has the same capacity to act as an adult as far as that business is concerned.
          Question: A juridical act by a minor who is permitted to conduct business may not be rescinded even if the judicial act does not concern the business. True or False? Let's think step by step 
          Answer: A juridical act in contravention of the provisions of the preceding paragraph is voidable. So the answer is False
          
          Article 6
          (1) A minor who is permitted to conduct one or multiple types of business has the same capacity to act as an adult as far as that business is concerned.
          (2) In a case as referred to in the preceding paragraph, if there are grounds that make the minor unable to sustain that business, the legal representative may revoke or limit the permission therefor in accordance with the provisions of Part IV (Relatives).
          Question: If a minor, after getting permission for carrying a business from his/her statutory agent, performs an act unrelated to the business without getting consent of the statutory agent, that act may not be rescinded. True or False? Let's think step by step 
          Answer: The minor's statutory agent may revoke or limit the permission therefor in accordance with the provisions of Part IV (Relatives). So the answer is False
          
          Article 6
          (1) A minor who is permitted to conduct one or multiple types of business has the same capacity to act as an adult as far as that business is concerned.
          (2) In a case as referred to in the preceding paragraph, if there are grounds that make the minor unable to sustain that business, the legal representative may revoke or limit the permission therefor in accordance with the provisions of Part IV (Relatives).
          Question: A minor who is permitted to carry on one or more kinds of business shall have the same capacity to act as a person of the age of majority as far as such business is concerned. True or False? Let's think step by step 
          Answer: Article 6 (1) A minor who is permitted to conduct one or multiple types of business has the same capacity to act as an adult as far as that business is concerned. So the answer is True
          
          Article 5 (1) A minor must obtain the consent of the minor's legal representative to perform a juridical act;provided, however, that this does not apply to a juridical act for merely acquiring a right or being released from an obligation.
          (2) A juridical act in contravention of the provisions of the preceding paragraph is voidable.
          Article 6 (1) A minor who is permitted to conduct one or multiple types of business has the same capacity to act as an adult as far as that business is concerned.
          Question: A juridical act by a minor who is permitted to conduct business may not be rescinded even if the judicial act does not concern the business. True or False?
          Answer: "
    ```
     
    =>Ví dụ về answer: "let's think step by step answer: a juridical act in contravention of the provisions of the preceding paragraph is voidable. so the answer is false"

  - next week: 
    - share code chung va thu nghiem 
    - [!] Optimize CoT prompting (variant prompting performance checking)
    - fine-tuning 7b LLM (e.g., LLama, FlanT5)
    - Constractive learning ? 


--- 
### Meeting on Jan 1, 2024
