

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
    - Few-Shot + CoT prompting ([t4_CoT_prompting_flow](t4_CoT_prompting_flow.MD)): 
      - Ap dụng thêm CoT cho cả 3 ví dụ trên xong ghép lại thành 3-shot cho statute query
      - **Result**:  TestR03: (76%-> 82%), TestR04: (78% -> 83%)

  - next week: 
    - share code chung va thu nghiem 
    - [!] Optimize CoT prompting (variant prompting performance checking)
    - fine-tuning 7b LLM (e.g., LLama, FlanT5)
    - Constractive learning ? 


--- 
### Meeting on Jan 4, 2024
- Task 1 (Minh):
  - Previous week
    - [!] Recall score of Bm25 va Tfidf (top 50, 80, 100, 150)?
    - [!] Su dung MonoT5 fine-tune cap relevant. => Su dung code train cua task 2.
    - BM25:
        - Query = chunk contains <FRAGMENT_SUPPRESSED>
    - GNN: inductive link prediction ?
    - Other method to increase Precision?
        - Finetune LLM to classify relevant
        - CoT LLM to construct the query
        - 
    - Push Recall higher. Method: BM25. Using different querying strategy
        - Query using all sentences. Each sentence -> K candidates.
          | K (sentences-based)| Coverage | Precision | Recall | F1 |
          | --- | --- | --- | --- | --- |
          | 10 | 0.1112 | 0.0154 | 0.8038 | 0.0303 |
          | 20 | 0.1946 | 0.0094 | 0.8555 | 0.0186 |
          | 50 | 0.3701 | 0.0053 | 0.9169 | 0.0105 |
          | 80 | 0.4869 | 0.0041 | 0.9422 | 0.0082 |
          | 120 | 0.5957 | 0.0034 | 0.9554 | 0.0068 |
          | 150 | 0.6548 | 0.0031 | 0.9699 | 0.0063 |
        - Query using chunks of sentences with length ~ 10000 characters:
          | K (sentences-based) | Coverage | Precision | Recall | F1 |
          | --- | --- | --- | --- | --- |
          | 50 | - | 0.0262 | 0.6570 | 0.0504 |
          | 80 | - | 0.0138 | 0.6955 | 0.0271 |
          | 120 | - | 0.0101 | 0.7352 | 0.0199 |
          | 150 | - | 0.0085 | 0.7533 | 0.0168 |
        - Query using fragment suppressed sentences (if exists, else using all sentences)
          | K (sentences-based) | Coverage | Precision | Recall | F1 |
          | --- | --- | --- | --- | --- |
          | 10 | 0.0427 | 0.0351 | 0.7027 | 0.0669 |
          | 20 | 0.0800 | 0.0203 | 0.7605 | 0.0395 |
          | 50 | 0.1723 | 0.0102 | 0.8279 | 0.0203 |
          | 80 | 0.2470 | 0.0074 | 0.8628 | 0.0148 |
          | 120 | 0.3289 | 0.0057 | 0.8844 | 0.0114 |
          | 150 | 0.3811 | 0.00502 | 0.8953 | 0.0100 |
        - Query using TF-IDF top 25 key tokens:
          | K | Coverage | Precision | Recall | F1 |
          | --- | --- | --- | --- | --- |
          | 10 | 0.0082 | 0.1090 | 0.4187 | 0.1730 |
          | 20 | 0.0164 | 0.0689 | 0.5294 | 0.1220 |
          | 50 | 0.0410 | 0.0349 | 0.6702 | 0.0663 |
          | 80 | 0.0657 | 0.0236 | 0.7256 | 0.0457 |
          | 120 | 0.0986 | 0.0165 | 0.7641 | 0.0324 |
          | 150 | 0.1232 | 0.0137 | 0.7906 | 0.0269 |
  - Next week: 
    - su dung rank top K theo document (top k most closed documents)
    - Using task 2 code, finetune MonoT5  (thu voi top 20, 10)


- Task 2 (Hiep):
  - Previous week
  - Next week 
    - step1: Fine-tuned Mono T5 1 query case, all case -> get top-k (k=10) for similar case. 
    - step2: Fine-tuned Mono T5 1 query case, k retrieved case from step 1 -> final model. 
    - Check relevance between base cases and entailed fragments
      - Try Rerank MotoT5 to search for relevant paragraphs in base cases


- Task 3 (Phuong, Dat) 
  - previous week: 
    - optimize hyper-parameters fine-tuned model Bert Japanese. 
    - Rerun mono-T5 for enssembling  
    - Result: 
        | Setting   |      R02 (dev)     |  R03 (test 1) |  R04 (test 2) |
        |----------|-------------:|------:|------:|
        | SOTA 2023 |  76.36| 85.52 | 75.69|
        | -- |      |
        | Baseline bertJp (!) (Mckpt) |  69.01   |   77.46   | 73.08 |
        | Baseline masked-whole-word (!) (Mckpt) |   66.85   |   78.01 | 76.97 |
        | combination2models (Mckpt) |   69.49  |   78.35 | 77.54 |
        | -- |     |
        | Mono T5 (zero)|    ?   |   ? | ? |
        | Mono T5 (ft)|    68.38   |   81.24 |  60.00 |
        | -- |     |
        | combination2models+MonoT5_lastyear  |   75.33  |   85.04 | 77.23 |
        | combination2models+MonoT5   |   75.24  | 84.55 | 77.15  |
  - next week: 
    - output of R05 and ranking top 100, (-L) 
    - prompting FlanT5 xxlarge (ghep cap infer) with fewshot. 
  

- Task 4 (Cong, An) 
  - previous week:
    - [!] Optimize CoT prompting (variant prompting performance checking)
    - last week result:
      - Ap dụng thêm CoT cho cả 3 ví dụ trên xong ghép lại thành 3-shot cho statute query
      - **Result**:  TestR03: (76%-> 82%), TestR04: (78% -> 83%)
    - result:
      Update kết quả theo 3-shot_CoT:
      - R01: 65% -> 72%
      - R02: 80% -> 85%
      - R03: 76% -> 83%
      - R04: 78% -> 88%

  - Next week
    - [!] thong ke chi tiet ket qua cua cac prompt va performance tuong ung. 
    - Constrastive learning, tim similar query trong setting fewshot sao cho can bang phan phoi cau tra loi: Answer True/False  

---
### Meeting on Jan 11, 2024

- Task 1 (Minh):
  - Previous week: 
    - su dung rank top K theo document (top k most closed documents)
      - on documents: K=150
        - Average # candidates: 150
          Precision: 0.014566353187042842
          Recall: 0.838748495788207
          F1: 0.028635401902179496
      - on document chunks:
        - K = 150
          Average # candidates: 59.172413793103445
          Precision: 0.03183937274846366
          Recall: 0.7232250300842359
          F1: 0.06099355558938448
        - K = 300
          Average # candidates: 113.83385579937304
          Precision: 0.018092694076501527
          Recall: 0.7906137184115524
          F1: 0.03537583458970493
    - Using task 2 code, finetune MonoT5  (thu voi top 20, 10) [PENDING]
  - next week:
    - chay ra ket qua step 2 voi monoT5
      - kq cua hien tai (F1 score) 
      - kq (+recall score, + F1 scores) cua top K= 30 40 50 
    - ghep cap q_document vs 1 document trong top K sd LLM (mistral.)
- Task 2 (Hiep)
  - previous work:
    - monoT5 chay ra 5 checkpoint 
  - next week:
    - rerank prompt su dung output cua MonoT5. (thu nghiem voi FlanT5-XXl va Mistral)  
    - thu nghiem nhieu prompt
- Task 3 (Phuong, Dat):
  - previous week:  
    - output of R05 and ranking top 100, (-L) 
    - prompting FlanT5 xxlarge (ghep cap infer) with fewshot (detail in [t3_prompting.MD](./t3_prompting.MD)). 
      - rerank prompting 
      - reasoning (CoT task 4) prompting 
    - Result: 
        | Setting   |      R02 (dev)     |  R03 (test 1) |  R04 (test 2) |
        |----------|-------------:|------:|------:|
        | SOTA 2023 |  76.36| 85.52 | 75.69|
        | -- |      |
        | Baseline bertJp (!) (Mckpt) |  69.01   |   77.46   | 73.08 |
        | Baseline masked-whole-word (!) (Mckpt) |   66.85   |   78.01 | 76.97 |
        | **combination2models (Mckpt)** |   69.49  |   78.35 | 77.54 |
        | -- |     |
        | Mono T5 (ft Large-10k, zero-3b)|    68.38   |   81.24 |  60.00 |
        | Mono T5 (prompting) v2 |   71.54   |  79.83 | 58.96  |
        | -- |     |
        | combination2models+MonoT5_lastyear  |   **75.33**  |   85.04 | 77.23 |
        | combination2models+MonoT5   |   75.24  | 84.55 | 77.15  |
        | **combination2models+MonoT5  v2**   |   76.13  | 85.17 | 78.23  |
        | -- |     |
        | combination2models => rerank LLM   |   66.67  | 78.87 | 75.44  |
        |  rerank LLM + monoT5   |   73.87 | 84.04 | 77.87  |
        |  **rerank LLM + monoT5  + combination2models** |   75.04 | **85.68** | **79.94**  |

  - next week: 
    - Dat qua task 4 
- Task 4 (Cong, An)
  - previous week:
    - [!] thong ke chi tiet ket qua cua cac prompt va performance tuong ung. 
    - Constrastive learning, tim similar query trong setting fewshot sao cho can bang phan phoi cau tra loi: Answer True/False  
  - next week: 
    - Thu voi LLM khac: Mistral 8x7b, thu voi prompt tot nhat cu.   
    - fine-tuning: FlanT5_xx => enssemble?  
    - fine-tuning chay vs input instruction cua CoT 
    - idea lm len vu tru 
