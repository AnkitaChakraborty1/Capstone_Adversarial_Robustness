from textattack import AttackArgs, Attacker
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
from textattack import Attack
from textattack.constraints.pre_transformation import (
            RepeatModification,
                StopwordModification,
                )
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapQWERTY
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapWordNet,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)
from textattack.constraints.overlap import LevenshteinEditDistance,MaxWordsPerturbed
#from textattack.attack_recipe import AttackRecipe

from abc import ABC, abstractmethod

from textattack import Attack

class Recipe(Attack,ABC):
    @staticmethod
    def build(model_wrapper):

        transformation = CompositeTransformation(
            [
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapEmbedding(),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                #WordSwapQWERTY(
                #    random_one=False, skip_first_char=True, skip_last_char=True
                #),
            ])
        constraints = [RepeatModification(), StopwordModification()]
         #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        constraints.append(LevenshteinEditDistance(30))
        constraints.append(MaxWordsPerturbed(max_percent=0.2, compare_against_original=True))

        #
        goal_function = UntargetedClassification(model_wrapper)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)
        
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
# We wrap the model so it can be used by textattack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

#dataset = HuggingFaceDataset("rotten_tomatoes", None, "test", shuffle=True)
dataset = HuggingFaceDataset("imdb", split="test")

attack = Recipe.build(model_wrapper)
attack_args = AttackArgs(num_examples=10,log_to_csv="log.csv",checkpoint_interval=5,checkpoint_dir="checkpoints")
attacker = Attacker(attack, dataset, attack_args)
attack_results = attacker.attack_dataset()

