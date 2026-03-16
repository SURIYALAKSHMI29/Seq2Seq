# Using BPE

    ## 1k samples
        Src vocab: 2k
        Trg vocab: 2k

        Unique words in Src: 3841
        Unique words in trg: 4779

        LR = 0.001, batch = 32, epochs = 10

        Average loss 7.267549395561218
        Token accuracy 0.02024584237165582
        Average BLEU 0.001500319842962967

    ## 5k samples

        Src vocab: 9k
        Trg vocab: 11k

        Unique words in Src: 12931
        Unique words in trg: 17588

        LR = 0.001, batch = 32, epochs = 10

        Average loss 8.86339819431305
        Token accuracy 0.053943739658025375
        Average BLEU 0.0015445672167233356

------------------------------------------------------------------------------------------------------------------------

# Without BPE

    ## 1k samples

        Unique words in Src: 3841
        Unique words in trg: 4779

        LR = 0.0001, batch = 32, epochs = 10

        Average loss 4.806384801864624
        Token accuracy 0.536697247706422
        Average BLEU 0.044920106182587835

    ## 10k samples
        Unique words in Src: 20584
        Unique words in trg: 29458

        LR = 0.001, batch = 32, epochs = 10

        Average loss 6.733721435070038
        Token accuracy 0.22486232918621252
        Average BLEU 0.010896155426583856
