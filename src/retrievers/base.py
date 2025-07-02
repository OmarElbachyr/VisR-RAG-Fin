from abc import ABC, abstractmethod
from typing import Dict
from src.evaluation.pytrec_evaluator import PyTrecEvaluator
from src.evaluation.ir_evaluator import IrMeasuresEvaluator

class BaseRetriever(ABC):
    def __init__(self):
        self.pytrec_evaluator = PyTrecEvaluator()
        self.ir_evaluator = IrMeasuresEvaluator()
    
    @abstractmethod
    def search(self, queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        """Search method to be implemented by child classes"""
        pass

    def evaluate(self, run: Dict[str, Dict[str, float]], 
                qrels: Dict[str, Dict[str, int]], 
                k_values: list = [1, 3, 5, 10],
                verbose: bool = True,
                eval_lib: str = 'ir_measures') -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval results
        Args:
            run: Dict[qid -> Dict[doc_id -> score]]
            qrels: Dict[qid -> Dict[doc_id -> relevance]]
            verbose: Whether to print results
            eval_lib: Which evaluation library to use ('pytrec_eval' or 'ir_measures'): default is 'ir_measures'
        Returns:
            Evaluation metrics for different k values
        """
        if eval_lib == 'pytrec_eval':
            return self.pytrec_evaluator.evaluate(run, qrels, k_values, verbose)
        elif eval_lib == 'ir_measures':
            return self.ir_evaluator.evaluate(run, qrels, k_values, verbose)
        else:
            raise ValueError(f"Unknown eval_lib: {eval_lib}")