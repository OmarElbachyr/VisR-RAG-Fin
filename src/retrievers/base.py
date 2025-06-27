from abc import ABC, abstractmethod
from typing import Dict
from evaluation.evaluator import Evaluator
from evaluation.evaluator_ir import Evaluator_ir

class BaseRetriever(ABC):
    def __init__(self):
        self.evaluator = Evaluator()
        self.evaluator_ir = Evaluator_ir()
    
    @abstractmethod
    def search(self, queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        """Search method to be implemented by child classes"""
        pass

    def evaluate(self, run: Dict[str, Dict[str, float]], 
                qrels: Dict[str, Dict[str, int]], 
                k_values: list = [1, 3, 5, 10],
                verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval results
        Args:
            run: Dict[qid -> Dict[doc_id -> score]]
            qrels: Dict[qid -> Dict[doc_id -> relevance]]
            verbose: Whether to print results
        Returns:
            Evaluation metrics for different k values
        """
        return self.evaluator.evaluate(run, qrels, k_values, verbose), self.evaluator_ir.evaluate(run, qrels, k_values, verbose)