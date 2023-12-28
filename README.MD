

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

Task 1 (Minh)

- Previous week:
Precision: 0.11526627218934911
Recall: 0.5860409145607701
F1: 0.19264240506329114
    - Method:
        - using BM25 to rank candidate:
            - Query = top 20 keyword TF-IDF
            - Corpus = whole document
        - Train GNN for link prediction
            - Node embedding: mean Legal-BERT sentence embeddings
    - Problem:
        - train documents $\cap$  test documents $= \empty$
- TODO next week:
    - BM25:
        - Query = chunk contains <FRAGMENT_SUPPRESSED>
    - GNN: inductive link prediction ?
    - Other method to increase Precision?
        - Finetune LLM to classify relevant
        - CoT LLM to construct the query

Task 3 (Phuong, Dat)

previous week:
SOTA 2023:
dev R02: 76.36
public test R03: 85.52
private test R04: 75.69
Baseline (Mckpt setting):
dev R02: 70.31
public test R03: 76.94
public test2 R04: 77.17

next week:
rerun mono-T5 for enssembling (doing -> fix bug confict)
features extraction by Llama-2 model. (todo)
summarize doc and re-ranking by LLMs. (todo)
