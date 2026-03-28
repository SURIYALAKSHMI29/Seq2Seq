# Seq2Seq Translation Model Report

## Dataset Summary
- Total number of samples: **1,35,842**
- Total pairs after filtering: **14,242**
- Source vocab size: **4,631**
- Target vocab size: **6,623**
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
- Average Loss: **2.786386**
- Token Accuracy: **0.519585**
- BLEU Score: **0.2559416**

### Sample Predictions

**Input:** i don't have any money.  
**Target:** je n'ai pas un `<unk>` `<unk>`  
**Predicted:** je n'ai pas blé.  

---

**Input:** are you brushing your teeth properly?  
**Target:** te brosses-tu bien les dents ?  
**Predicted:** te brosses-tu les les dents ? ? 

---

**Input:** i can't stand listening to <unk> music.
**Target:** je ne supporte pas <unk> de la musique <unk>
**Predicted:** je ne pas à les les les les les 

---

**Input:** i think you're jealous.  
**Target:** je pense que vous êtes jaloux.  
**Predicted:** je pense que tu es jalouse.  

---

**Input:** i want one, too.  
**Target:** j'en veux une aussi.  
**Predicted:** j'en veux un aussi.  

---

## Training Performance
- Average Loss: **1.115121**
- Token Accuracy: **0.691473**
- BLEU Score: **0.449588**

### Sample Predictions

**Input:** i don't recognize any of the people in the picture. 
**Target:** je ne reconnais aucune des personnes sur la photo. 
**Predicted:** je ne reconnais aucune des personnes personnes ces photo.

---

**Input:** are you allergic to anything else?
**Target:** est-ce que vous êtes allergiques à quelque chose d'autre ?
**Predicted:** est-ce que à êtes quelque chose ? ? ?

---

**Input:** do you know what tom was wearing?
**Target:** sais-tu ce que tom portait ?
**Predicted:** sais-tu ce que tom portait ?

---

**Input:** i don't like the way you laugh at her.
**Target:** je n'aime pas la façon dont tu te moques d'elle. 
**Predicted:** je n'aime pas la façon dont vous vous d'elle. 

---

**Input:** i want to make you happy.
**Target:** je veux te rendre heureuse.
**Predicted:** je veux te rendre heureuse.

---
