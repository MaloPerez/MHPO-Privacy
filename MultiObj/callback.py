"""SMAC callbacks for wandb trial logging during HPO runs."""

import smac
from smac import Callback
from smac.runhistory import TrialInfo, TrialValue
import wandb


class CustomCallback(Callback):
    def __init__(self) -> None:
        self.trials_counter = 0
        self.adv_found = False

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue):
        self.trials_counter += 1
        print(f"Evaluated {self.trials_counter} trials so far.")
        wandb.log({"Trials": self.trials_counter})

        if self.trials_counter % 10 == 0:
            smbo.print_stats()

        return None


class CustomCallback_Adv(Callback):
    def __init__(self) -> None:
        self.trials_counter = 0

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue):
        self.trials_counter += 1
        print(f"Evaluated {self.trials_counter} trials so far.")
        # wandb.log({"Adv Search Trials": self.trials_counter})

        if self.trials_counter % 10 == 0:
            smbo.print_stats()

        return None