# Capstone_Adversarial_Robustness
Exploring Robustness using TextAttack
I.	Introduction
An area that has been gaining impetus in the field of applied artificial intelligence is adversarial attacks. In this kind of attacks, the input data of a neural network is manipulated deliberately in order to verify its resistance to delivering the same outputs.
In adversarial word-substitution attacks for neural NLP models there has been an advent of numerous models over the past few years. In this project I will conduct analysis of robustness of various attacks in NLP domain and formulate ways to alter the factor of “Robustness” leading to more diverse adversarial text attacks.  This work heavily relies on TextAttack, (a Python framework for adversarial attacks, data augmentation, and adversarial training in NLP) for deducing the robustness of various models under attack from pre-existing or fabricated attacks. 

II.	Evaluation Metrics 
There are 4 basic metrics to evaluate the robustness of defenders
i.	Clean accuracy: classification accuracy of the model on the clean test dataset.
ii.	Accuracy under attack: prediction accuracy of the model under specific adversarial attack methods. 
iii.	Attack success rate: the number of texts that an attack algorithm successfully manipulates over the number of all texts attempted. 
iv.	Number of Queries: the average number of times the model is queried by the attacker. The idea is that the greater the average number of queries needed for attacker, the harder it is to compromise the defense model.
So, to be a called a good/robust defense method it should have the following
i.	higher clean accuracy
ii.	higher accuracy under attack
iii.	lower attack success rate
iv.	requires larger number of queries for attack.

III.	Frameworks and libraries used
Textattack - Providing implementations of 16 adversarial attacks from the literature, TextAttack supports a varied models and datasets, including BERT and other transformers. It is also inclusive of data augmentation and adversarial training modules for using components of adversarial attacks designed for improvement of model accuracy and robustness. TextAttack is democratizing NLP: anyone can try data augmentation and adversarial training on any model or dataset, with just a few lines of code.

Huggingface - A community and NLP platform providing users with access to a wide variety of tools aiding them in boosting language-related workflows. There are thousands of models and datasets as a part of Huggingface framework to enable data scientists and machine learning engineers alike to tackle tasks such as text classification, text translation, text summarization, question answering, or automatic speech recognition. In this project I plan to use the pretrained models offered by HuggingFace for example BERT, DistilBERT, GPT2, or T5, to name a few.
In this example I have used the albert base v2 model from HuggingFace. This model was trained with data from imbd, a set of movie reviews with either positive or negative labels.
 

