
- Total number of samples: **1,35,842**
- Total pairs after filtering: **14,242**
- Source vocab size: **6,623** 
- Target vocab size: **4,631**
- Train samples: **11,393**
- Validation samples: **2,849**


epochs = 5, lr = 0.001, 

**Average loss 3.0676361931694878**
**Token accuracy 0.5912116510685151**
**Average BLEU 0.4128136756719476**

### Samples

Input    : je ne pense pas que ça va arriver.
Target   : i don't think that's going to happen.
Predicted: i don't think it'll will work.

----------------------------------------------------------------------------------------------------

Input    : j'ai une <unk>
Target   : i have an <unk>
Predicted: i have a receipt.

---------------------------------------------------------------------------------------------------- 

Input    : je pense que nous pouvons gérer ça.
Target   : i think we can handle this.
Predicted: i think we can fix this.

---------------------------------------------------------------------------------------------------- 

Input    : il fait les préparatifs pour un voyage.
Target   : he is making <unk> for a trip.
Predicted: he is working for a trip.

---------------------------------------------------------------------------------------------------- 

Input    : je ne peux pas vous permettre de faire ça.
Target   : i can't let you do that.
Predicted: i can't afford to do it anymore.

---------------------------------------------------------------------------------------------------- 

Input    : j'ai <unk> le train.
Target   : i was late for the train.
Predicted: i have a weird neighbor.

---------------------------------------------------------------------------------------------------- 

Input    : je veux te plaire.
Target   : i want to please you.
Predicted: i want to see you laugh.