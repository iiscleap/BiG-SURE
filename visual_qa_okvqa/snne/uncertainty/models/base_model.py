from abc import ABC, abstractmethod
from typing import List, Text


STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']
VLLM_STOP_SEQUENCES = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:", 
                   "\n\nProblem", "\nProblem", "Problem:", "<|eot_id|>", "####"]

class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass
