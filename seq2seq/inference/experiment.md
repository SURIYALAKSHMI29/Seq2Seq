
Batch = 64, lr = 0.001, optimizer = Adam, epochs = 10

## FRENCH TO ENGLISH

Total number of samples: 135842
prefixes [('je', 'ne'), ('je', 'suis'), ('je', 'veux'), ('il', 'est'), ('il', 'a'), ('je', 'me'), ('je', "n'ai"), ('je', 'pense'), ('tom', 'a'), ('vous', 'êtes')]

After filtering, total num of pairs 18236
train samples 14588
test samples 3648
Vocab size 8043
Vocab size 5417
il est fiancé à ma sœur. <SOS> he is engaged to my sister. <EOS>
je suis plus solide que tu ne le penses. <SOS> i'm tougher than you think. <EOS>

src vocab size 8071
trg vocab size 5438
Model instantiated
Model is on device: cuda:0
CUDA available: True
Epoch 0  |  Train 4.986948557067335  |  Val 3.799904702002542
Epoch 1  |  Train 3.936325726801889  |  Val 3.214066271196332
Epoch 2  |  Train 3.340361189423946  |  Val 2.881057705795556
Epoch 3  |  Train 2.9269389614724277  |  Val 2.702915488627919
Epoch 4  |  Train 2.601921405708581  |  Val 2.5886065374340927
Epoch 5  |  Train 2.402697470627333  |  Val 2.5335270626503124
Epoch 6  |  Train 2.221883553161956  |  Val 2.5088770807835092
Epoch 7  |  Train 2.122975109961995  |  Val 2.511843332073145
Epoch 8  |  Train 2.0067294867415177  |  Val 2.500568624128375
Epoch 9  |  Train 1.956463556017792  |  Val 2.509104281141047
Time taken 76.20922088623047

**Average loss 2.5091034466760203**
**Token accuracy 0.5883446993180409**
**Average BLEU 0.3450380014421805**

Input    : je ne le tolèrerai pas.
Target   : i will not stand for this.
Predicted: i didn't tolerate it.

Input    : je me joindrai à toi.
Target   : i'll come with you.
Predicted: i'll join with you.

Input    : je ne le comprends pas.
Target   : i don't understand it.
Predicted: i do not understand it.



## ENGLISH TO FRENCH

Total number of samples: 135842
prefixes [('je', 'ne'), ('je', 'suis'), ('je', 'veux'), ('il', 'est'), ('il', 'a'), ('je', 'me'), ('je', "n'ai"), ('je', 'pense'), ('tom', 'a'), ('vous', 'êtes')]

After filtering, total num of pairs 18236
train samples 14588
test samples 3648
Vocab size 5416
Vocab size 8042
it's impossible to anticipate every possible situation. <SOS> il est impossible d'anticiper toutes les situations possibles. <EOS>
i no longer believe it. <SOS> je ne le crois plus. <EOS>


src vocab size 5416
trg vocab size 8042
Model instantiated
Model is on device: cuda:0
CUDA available: True
Epoch 0  |  Train 4.939684683816475  |  Val 3.512691623286197
Epoch 1  |  Train 3.8747044299778186  |  Val 3.0519749825460867
Epoch 2  |  Train 3.3088041732185767  |  Val 2.734263035289028
Epoch 3  |  Train 2.934010491036532  |  Val 2.6194374854104563
Epoch 4  |  Train 2.6558103456831814  |  Val 2.5256707793787907
Epoch 5  |  Train 2.4950506258429144  |  Val 2.4890280941076446
Epoch 6  |  Train 2.363986241712905  |  Val 2.4586688008224753
Epoch 7  |  Train 2.287557251097863  |  Val 2.411532820316783
Epoch 8  |  Train 2.2212443963477484  |  Val 2.423857835301182
Epoch 9  |  Train 2.1553429093277243  |  Val 2.4071447995671056
Time taken 91.83284997940063


**Average loss 2.4071436911298516**
**Token accuracy 0.6103315176220194**
**Average BLEU 0.34290644448357044**