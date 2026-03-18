# Using BPE

## 1k samples
Src vocab: 2k | Trg vocab: 2k

Unique words in Src: 3841

Unique words in trg: 4779

LR = 0.001, batch = 32, epochs = 10

*Average loss 7.267549395561218*

*Token accuracy 0.02024584237165582*

*Average BLEU 0.001500319842962967*

 ## 5k samples

Src vocab: 9k | Trg vocab: 11k

Unique words in Src: 12931

Unique words in trg: 17588

LR = 0.001, batch = 32, epochs = 10

*Average loss 8.86339819431305*

*Token accuracy 0.053943739658025375*

*Average BLEU 0.0015445672167233356*

---

# Without BPE

## 1k samples

Unique words in Src: 3841

Unique words in trg: 4779

LR = 0.0001, batch = 32, epochs = 10

*Average loss 4.806384801864624*

*Token accuracy 0.536697247706422*

*Average BLEU 0.044920106182587835*

## 10k samples
Unique words in Src: 20584

Unique words in trg: 29458

LR = 0.001, batch = 32, epochs = 10

*Average loss 6.733721435070038*

*Token accuracy 0.22486232918621252*

*Average BLEU 0.010896155426583856*


# With dropout

## 1k samples

Average loss 5.603278636932373

Token accuracy 0.5348623853211009

Average BLEU 0.04496510985992143

---

# english to french dataset

src vocab size 21573 | trg vocab size 35434

## 10 epochs, 128 batch size, lr = 0.001

Average loss 5.396249764402148

Token accuracy 0.2828987438142368

Average BLEU 0.006831448289986097

## 5 epochs, 128 batch size, lr = 0.001

Average loss 5.212172517194435

Token accuracy 0.2764084507042254

Average BLEU 0.006987012549998677


# Using filtered dataset

src vocab size 4631 | trg vocab size 6623

After filtering, total num of pairs 14242

train samples 11,393

test samples 2,849

## epochs = 5, batch = 128, lr = 0.001

Average loss 3.825610658396845

Token accuracy 0.34665116954816216

Average BLEU 0.03659988053008612

## epochs = 10, batch = 128, lr = 0.001

Average loss 4.143194405928902

Token accuracy 0.34277688992203015

Average BLEU 0.0371200709565255

## epochs = 50, batch = 128, lr = 0.001

Average loss 5.889262012813402

Token accuracy 0.34185674851082376

Average BLEU 0.037273301050900355

## epochs = 50, batch = 128, lr = 0.0001

Average loss 4.097996494044429

Token accuracy 0.32413191922126977

Average BLEU 0.03143145739089868

## 2 optimizers

Average loss 7.986370998880138

Token accuracy 0.21555523269892005

Average BLEU 0.03076839976343679

## epochs = 5, lr = 0.001, batch = 64, 2 diff dropouts (0.2, 0.3)

Average loss 3.8282476319207084

Token accuracy 0.3410334640902707

Average BLEU 0.03711476352650673

## epochs = 10, lr = 0.001, b = 64

Average loss 4.111740271250407

Token accuracy 0.3560462976415323

Average BLEU 0.041113587628570154

## epochs = 10, lr = 0.001, b = 64, 2 diff dropouts (0.2, 0.4)

Average loss 4.101969284481473

Token accuracy 0.3500895927163543

Average BLEU 0.037883483091174716

## epochs = 10, lr = 0.0001, b = 64, 2 diff dropouts (0.2, 0.4)

Average loss 4.307830842336019

Token accuracy 0.32897476875393483

Average BLEU 0.030796324017063677

## epochs = 15, lr = 0.0001, b = 64, 2 diff dropouts (0.2, 0.4)

Average loss 4.102601321538289

Token accuracy 0.3385636108286115

Average BLEU 0.03310089937152901

## epochs = 20, lr = 0.0001, b = 64, 2 diff dropouts (0.2, 0.4)

Average loss 4.061820485856798

Token accuracy 0.33488304518378614

Average BLEU 0.03285951074717311


## batch = 64, lr = 0.001, tf = 0.7, dp = (0.1, 0.2), embed = 128, hidden = 128

Average loss 2.051729290845008

Token accuracy 0.5300200851226627

Average BLEU 0.12331225409459133

## batch = 64, lr = 0.001, tf = 0.6, dp = (0.1, 0.2), embed = 128, hidden = 128

Average loss 1.9367405685632588

Token accuracy 0.5451437042704796

Average BLEU 0.12711502528903332

## batch = 64, lr = 0.001, tf = 0.5, dp = (0.1, 0.2), embed = 128, hidden = 128

Average loss 1.9239421956366

Token accuracy 0.5634952895605184

Average BLEU 0.13620502232228446


## batch = 64, lr = 0.001, tf = 0.6, dp = (0.1, 0.2), embed = 128, hidden = 256

Average loss 2.482236533694797

Token accuracy 0.49900721584580365

Average BLEU 0.11664961000499968


## batch = 64, lr = 0.001, tf = 0.6, dp = (0.1, 0.2), embed = 256, hidden = 512

Average loss 2.823939079708523

Token accuracy 0.5192527096406161

Average BLEU 0.23774996473850357


## batch = 64, tf = 0.6, dp = (0.1, 0.2), embed = 256, hidden = 512, 10 epoch lr = 0.001, nxt 5 epochs lr = 0.0001

Average loss 2.699438524246216

Token accuracy 0.5426411865373645

Average BLEU 0.2627080773452708

## batch = 64, tf = 0.6, dp = (0.1, 0.2), embed = 256, hidden = 512, 10 epoch lr = 0.001, nxt 10 epochs lr = 0.0001

Average loss 2.745206011666192

Token accuracy 0.5467769537934969

Average BLEU 0.27048012910656216