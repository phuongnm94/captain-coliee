
## Cách xây dựng fewshot + CoT:
### Step1: Sinh ra chain tự động từ tập train theo mẫu prompt "Let's think step by step why the answer is ..."

Ví dụ: 
```
    "Article 5
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
```
    Article 5 (1) A minor must obtain the consent of the minor's legal representative to perform a juridical act;provided, however, that this does not apply to a juridical act for merely acquiring a right or being released from an obligation.
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
     
Ví dụ về answer: 
```
"let's think step by step answer: a juridical act in contravention of the provisions of the preceding paragraph is voidable. so the answer is false"
```