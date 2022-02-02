from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack import Attack
from textattack import AttackArgs
from textattack.datasets import Dataset
from textattack.datasets import HuggingFaceDataset
from textattack import Attacker
from textattack.attack_recipes import TextFoolerJin2019,Seq2SickCheng2018BlackBox,PSOZang2020,MorpheusTan2020,HotFlipEbrahimi2017,GeneticAlgorithmAlzantot2018,FasterGeneticAlgorithmJia2019
from textattack.attack_recipes import BAEGarg2019,DeepWordBugGao2018,IGAWang2019,InputReductionFeng2018
from textattack.attack_recipes import Kuleshov2017,Pruthi2019,PWWSRen2019,TextBuggerLi2018
from textattack.loggers import CSVLogger # tracks a dataframe for us.
# https://huggingface.co/textattack
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
# We wrap the model so it can be used by textattack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

#dataset = HuggingFaceDataset("rotten_tomatoes", None, "test", shuffle=True)
dataset = HuggingFaceDataset("imdb", split="test")
#recipe_list = [BAEGarg2019,DeepWordBugGao2018]
#recipe_list = [InputReductionFeng2018]
#recipe_list = [TextFoolerJin2019,Seq2SickCheng2018BlackBox,
#recipe_list = [PSOZang2020,MorpheusTan2020,HotFlipEbrahimi2017,GeneticAlgorithmAlzantot2018,
recipe_list = [FasterGeneticAlgorithmJia2019]
#recipe_list = [Pruthi2019,PWWSRen2019,TextBuggerLi2018]
for recipe in recipe_list:
    print("Starting attack for-----------------",recipe)
    attack = recipe.build(model_wrapper)

    attack_args = AttackArgs(num_examples=10,log_to_csv="log.csv",checkpoint_interval=5,checkpoint_dir="checkpoints",disable_stdout=True)

    attacker = Attacker(attack, dataset, attack_args)

    attack_results = attacker.attack_dataset()
