# Seq2Seq Translation Model Report

## Dataset Summary
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

## Training Performance
- Average Loss: **1.1618**
- Token Accuracy: **0.6823**
- BLEU Score: **0.4423**

### Sample Predictions

**Input:** are you sure you don't want to come tonight?  
**Target:** es-tu sûre de ne pas vouloir venir ce soir ?  
**Predicted:** êtes-vous sûr de ne pas vouloir venir ce soir  

---

**Input:** do you study english?  
**Target:** étudiez-vous l'anglais ?  
**Predicted:** étudies-tu l'anglais ?  

---

**Input:** i don't see how that would help.  
**Target:** je ne vois pas en quoi ça aiderait.  
**Predicted:** je ne vois pas comment quoi aiderait. aiderait.  

---

**Input:** i want you to keep your eyes open.  
**Target:** je veux que tu gardes les yeux ouverts.  
**Predicted:** je veux que tu gardiez les yeux ouverts.  

---

**Input:** i'm not giving up.
**Target:** je n'abandonne pas.
**Predicted:** je n'abandonne n'abandonne pas 

---

## Validation Performance
- Average Loss: **2.8239**
- Token Accuracy: **0.5193**
- BLEU Score: **0.2683**

### Sample Predictions

**Input:** i don't have any money.  
**Target:** je n'ai pas un `<unk>` `<unk>`  
**Predicted:** je n'ai aucun blé.  

---

**Input:** are you brushing your teeth properly?  
**Target:** te brosses-tu bien les dents ?  
**Predicted:** te brosses-tu correctement correctement dents ?  

---

**Input:** i can't stand listening to <unk> music.
**Target:** je ne supporte pas <unk> de la musique <unk>
**Predicted:** je ne supporte pas supporter de de la  

---

**Input:** i think you're jealous.  
**Target:** je pense que vous êtes jaloux.  
**Predicted:** je pense que tu es jalouse.  

---

**Input:** i want one, too.  
**Target:** j'en veux une aussi.  
**Predicted:** j'en veux un aussi.  

---
