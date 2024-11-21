from SAC_for_carla_v2.utils import *
import numpy as np
import typing

"""
copyright : https://colab.research.google.com/github/davidrpugh/stochastic-expatriate-descent/blob/2020-04-14-prioritized-experience-replay/_notebooks/2020-04-14-prioritized-experience-replay.ipynb#scrollTo=QhNtCttOkScA


"""


class PER_Buffer:
    def __init__(self, opt,
                 device='cuda:0'):  # opt is option argument in main file. device for gpu memory setting buffer content to cpu
        self.device = device

        self.buffer_size = opt.buffer_size
        self.buffer_length = 0
        self.batch_size = opt.batch_size

        self.buffer = np.empty(self.buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])

    def __len__(self) -> int:
        return self.buffer_length

    def is_empty(self) -> bool:
        return self.buffer_length == 0

    def is_full(self) -> bool:
        return self.buffer_length == self.buffer_size

    def add(self, experience: Experience) -> None:

        "Add state,action reward, done,priority to replay buffer"

        # Set prior 1 if empty
        priority = 1.0 if self.is_empty() else self.buffer["priority"].max()

        if self.is_full():
            # 가장 작은 값과 바꾼다
            if priority > self.buffer["priority"].min():
                idx = self.buffer["priority"].argmin()
                self.buffer[idx] = (priority, experience)

            else:
                pass  # low prior not include in buffer
        else:
            self.buffer[self.buffer_length] = (priority, experience)
            self.buffer_length += 1

    def sample(self, beta,alpha) -> typing.Tuple[np.array, np.array, np.array]:

        ps = self.buffer[:self.buffer_length]["priority"]

        sampling_probs = ps ** alpha / np.sum(ps ** alpha)  # stochastic priority
        sampling_probs = np.nan_to_num(sampling_probs,nan=0.0)

        if np.sum(sampling_probs) > 0:
            sampling_probs = sampling_probs / np.sum(sampling_probs)
        else:
            print("\nAll probality became zero after handling nan val\n")
            sampling_probs = (ps + 1e-6) / np.sum(ps + 1e-6)

            #raise ValueError("All probality became zero after handling nan val")

        try:

            idxs = np.random.choice(np.arange(ps.size), size=self.batch_size, replace=True, p=sampling_probs)
        except ValueError as e:
            print(f"An error occurred in np.random.choice: {e}")
            print(f"ps.size: {ps.size}, sampling_probs.size: {sampling_probs.size}, sum: {np.sum(sampling_probs)}")

        # Select experience & compute IS
        experiences = self.buffer["experience"][idxs]

        IS = (self.buffer_length * sampling_probs[idxs]) ** -beta
        normalized_weights = IS / IS.max()

        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self.buffer["priority"][idxs] = priorities
