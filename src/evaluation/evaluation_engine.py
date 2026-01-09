"""  
src/evaluation/evaluation_engine.py  
-----------------------------------  
Comprehensive evaluation engine for RAG systems.  
"""  
  
import logging  
from typing import Dict, List, Any, Optional  
from datetime import datetime  
  
from src.evaluation.evaluator import Evaluator, EvaluationResult  
  
logger = logging.getLogger(__name__)  
  
  
class EvaluationEngine:  
    """  
    Comprehensive evaluation engine for RAG pipelines.  
      
    Provides:  
    - Batch evaluation  
    - Metric aggregation  
    - Evaluation history  
    - Performance tracking  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the evaluation engine.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self.evaluator = Evaluator(config)  
        self._evaluation_history: List[Dict[str, Any]] = []  
      
    def evaluate_response(  
        self,  
        query: str,  
        answer: str,  
        context: List[Dict[str, Any]],  
        reference_answer: Optional[str] = None,  
        metadata: Optional[Dict[str, Any]] = None  
    ) -> Dict[str, Any]:  
        """  
        Evaluate a single RAG response.  
          
        Args:  
            query: User query  
            answer: Generated answer  
            context: Retrieved documents  
            reference_answer: Optional ground truth  
            metadata: Optional metadata  
              
        Returns:  
            Evaluation results  
        """  
        # Run evaluation  
        results = self.evaluator.evaluate(  
            query=query,  
            answer=answer,  
            context=context,  
            reference_answer=reference_answer  
        )  
          
        # Calculate aggregate score  
        scores = [r.score for r in results.values()]  
        aggregate_score = sum(scores) / len(scores) if scores else 0.0  
          
        # Build result  
        evaluation = {  
            "timestamp": datetime.now().isoformat(),  
            "query": query,  
            "answer_length": len(answer),  
            "context_count": len(context),  
            "metrics": {name: r.to_dict() for name, r in results.items()},  
            "aggregate_score": aggregate_score,  
            "metadata": metadata or {}  
        }  
          
        # Store in history  
        self._evaluation_history.append(evaluation)  
          
        return evaluation  
      
    def evaluate_batch(  
        self,  
        test_cases: List[Dict[str, Any]]  
    ) -> Dict[str, Any]:  
        """  
        Evaluate a batch of test cases.  
          
        Args:  
            test_cases: List of test case dicts with query, answer, context  
              
        Returns:  
            Batch evaluation results  
        """  
        results = []  
          
        for case in test_cases:  
            result = self.evaluate_response(  
                query=case.get("query", ""),  
                answer=case.get("answer", ""),  
                context=case.get("context", []),  
                reference_answer=case.get("reference_answer"),  
                metadata=case.get("metadata")  
            )  
            results.append(result)  
          
        # Aggregate metrics  
        metric_scores = {}  
        for result in results:  
            for metric, data in result["metrics"].items():  
                if metric not in metric_scores:  
                    metric_scores[metric] = []  
                metric_scores[metric].append(data["score"])  
          
        metric_averages = {  
            metric: sum(scores) / len(scores)  
            for metric, scores in metric_scores.items()  
        }  
          
        return {  
            "total_cases": len(test_cases),  
            "results": results,  
            "metric_averages": metric_averages,  
            "overall_score": sum(metric_averages.values()) / len(metric_averages) if metric_averages else 0.0  
        }  
      
    def get_evaluation_history(  
        self,  
        limit: Optional[int] = None  
    ) -> List[Dict[str, Any]]:  
        """Get evaluation history."""  
        if limit:  
            return self._evaluation_history[-limit:]  
        return self._evaluation_history  
      
    def get_performance_summary(self) -> Dict[str, Any]:  
        """Get summary of evaluation performance."""  
        if not self._evaluation_history:  
            return {"message": "No evaluations recorded"}  
          
        # Aggregate by metric  
        metric_scores = {}  
        for eval_result in self._evaluation_history:  
            for metric, data in eval_result.get("metrics", {}).items():  
                if metric not in metric_scores:  
                    metric_scores[metric] = []  
                metric_scores[metric].append(data["score"])  
          
        summary = {  
            "total_evaluations": len(self._evaluation_history),  
            "metrics": {}  
        }  
          
        for metric, scores in metric_scores.items():  
            summary["metrics"][metric] = {  
                "mean": sum(scores) / len(scores),  
                "min": min(scores),  
                "max": max(scores),  
                "count": len(scores)  
            }  
          
        return summary  