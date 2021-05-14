# CMP5130 - Machine Learning and Pattern Recognition

I implemented naive bayes classifier with different continious variable density estimator options. Here are the results with them:


1) Assume gaussian distribution for continuous features. Report the accuracies for each of the following case:

  1.1) 100 samples for training, and rest for validation set 
	Train Accuracy:  0.59
	Number of Missclasification(Train):  41
	Train CM:  	
        [[20  3  0]
				 [13 31 15]
				 [ 1  9  8]]
	
	Test Accuracy:  0.5685553102771645
	Number of Missclasification(Test):  1759
	Test CM:  	
        [[ 759   56    1]
				 [ 695 1258  366]
				 [ 110  531  301]]
  
  1.2) 1000 samples for training, and rest for validation set
	Train Accuracy:  0.544
	Number of Missclasification(Train):  456
	Train CM:  	[[171  25   4]
				 [121 246 198]
				 [ 13  95 127]]
	
	Test Accuracy:  0.546742209631728
	Number of Missclasification(Test):  1440
	Test CM:  	[[573  64   2]
				 [394 776 643]
				 [ 55 282 388]]


2) Use Naive Estimator for each of the continuous feature. Determine the bin size for each feature with your own criterion. Report the accuracies for each of the following case:

  2.1) 100 samples for training, and rest for validation set
	Train Accuracy:  0.66
	Number of Missclasification(Train):  34
	Train CM:  	[[18  5  0]
				 [14 45  0]
				 [ 1 14  3]]
	
	Test Accuracy:  0.6068187392690704
	Number of Missclasification(Test):  1603
	Test CM:  	[[ 704  112    0]
				 [ 559 1717   43]
				 [  64  825   53]]
  
  
  2.2) 1000 samples for training, and rest for validation set
	Train Accuracy:  0.635
	Number of Missclasification(Train):  365
	Train CM:  	[[163  37   0]
				 [ 99 426  40]
				 [  5 184  46]]
	
	Test Accuracy:  0.6373937677053825
	Number of Missclasification(Test):  1152
	Test CM:  	[[ 542   97    0]
				 [ 326 1340  147]
				 [  34  548  143]]

3) Use Kernel Estimator for each of the continuous feature. Determine the bin size for each feature with your own criterion. Report the accuracies for each of the following case:

  3.1) 100 samples for training, and rest for validation set
	Train Accuracy:  0.65
	Number of Missclasification(Train):  35
	Train CM:  	[[18  5  0]
				 [14 45  0]
				 [ 1 15  2]]
	
	Test Accuracy:  0.6127054206524405
	Number of Missclasification(Test):  1579
	Test CM:  	[[ 695  121    0]
				 [ 529 1760   30]
				 [  56  843   43]]
  
  
  3.2) 1000 samples for training, and rest for validation set
	Train Accuracy:  0.634
	Number of Missclasification(Train):  366
	Train CM:  	[[162  38   0]
				 [ 96 433  36]
				 [  4 192  39]]
	
	Test Accuracy:  0.6405413912496065
	Number of Missclasification(Test):  1142
	Test CM:  	[[ 538  101    0]
				 [ 313 1371  129]
				 [  33  566  126]]




The heighest accuracy is at Case 3.2. Kernel estimator with 1000 training sample is the best.


