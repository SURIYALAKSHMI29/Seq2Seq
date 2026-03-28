# Seq2Seq Translation Model Report

## Dataset Summary
- Total number of samples: **1,35,842**
- Total pairs after filtering: **14,242**
- Source vocab size: **6,623** 
- Target vocab size: **4,631**
- Train samples: **11,393**
- Validation samples: **2,849**

---

## Training Configuration
- Epochs: **10**
- Learning rate: **0.001**
- Batch size: **64**
- Teacher forcing ratio: **0.6**

---

## Model Configuration
- Embedding size: **256**
- Hidden size: **512**
- Dropout:
  - Embedding: **0.1**
  - Fully connected: **0.2**

---

## Validation Performance
- Average Loss: **2.062478**
- Token Accuracy: **0.672488**
- BLEU Score: **0.446921**

### Sample Predictions

**Input:** je ne pense pas que ça va arriver. 
**Target:** i don't think that's going to happen. 
**Predicted:** i don't think it'll helping. it. 

---

**Input:** j'ai une lettre qu'il a <unk> 
**Target:** i have a letter written by him.
**Predicted:** i have a a for 

---

**Input:** j'ai été déçu par votre papier.
**Target:** i was disappointed with your paper.
**Predicted:** i was disappointed with your paper.

---

**Input:** j'étais <unk> 
**Target:** i was frightened. 
**Predicted:** i was careless. 

---

**Input:** je ne peux pas vous permettre de faire ça.
**Target:** i can't let you do that. 
**Predicted:** i can't afford you do that.

---

## Training Performance
- Average Loss: **0.645386**
- Token Accuracy: **0.829292**
- BLEU Score: **0.678876**

### Sample Predictions

**Input:** je fus aussi surpris que vous. 
**Target:** i was as surprised as you. 
**Predicted:** i was as surprised as you. you. 

---

**Input:** je ne faisais que plaisanter.
**Target:** i was just teasing. 
**Predicted:** i was just fooling.

---

**Input:** je ne veux pas vraiment vous voir souffrir.
**Target:** i don't really want to see you suffer. 
**Predicted:** i don't really really to see see suffer.  

---

**Input:** je ne vais pas bien.  
**Target:** i'm not well.
**Predicted:** i'm not sleeping well. well. well. well.   

---

**Input:** je dois vous parler immédiatement.
**Target:** i have to speak with you immediately.
**Predicted:** i have to talk to you immediately.

---
