"""  
src/evaluation/evaluator.py  
---------------------------  
RAG response evaluation metrics.  
"""  
  
import logging  
from typing import Dict, List, Any, Optional  
from dataclasses import dataclass  
import re  
  
logger = logging.getLogger(__name__)  
  
  
@dataclass  
class EvaluationResult:  
    """Result of an evaluation."""  
    metric: str  
    score: float  
    details: Dict[str, Any]  
      
    def to_dict(self) -> Dict[str, Any]:  
        return {  
            "metric": self.metric,  
            "score": self.score,  
            "details": self.details  
        }  
  
  
class Evaluator:  
    """  
    Evaluates RAG responses using multiple metrics.  
      
    Metrics:  
    - Relevance: How relevant is the answer to the query  
    - Faithfulness: Is the answer grounded in the context  
    - Completeness: Does the answer address all parts of the query  
    - Coherence: Is the answer well-structured and clear  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the evaluator.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self._llm = None  
        self._initialized = False  
      
    def _ensure_initialized(self):  
        """Lazy initialization of LLM for evaluation."""  
        if self._initialized:  
            return  
          
        try:  
            from langchain_openai import AzureChatOpenAI  
              
            azure_config = self.config.get_azure_openai_config()  
              
            if azure_config.get("azure_endpoint") and azure_config.get("api_key"):  
                self._llm = AzureChatOpenAI(  
                    azure_endpoint=azure_config["azure_endpoint"],  
                    api_key=azure_config["api_key"],  
                    api_version=azure_config["api_version"],  
                    azure_deployment=azure_config["azure_deployment"],  
                    temperature=0.0,  
                    max_tokens=500  
                )  
              
            self._initialized = True  
              
        except Exception as e:  
            logger.error(f"Failed to initialize evaluator LLM: {e}")  
            self._initialized = True  
      
    def evaluate(  
        self,  
        query: str,  
        answer: str,  
        context: List[Dict[str, Any]],  
        reference_answer: Optional[str] = None  
    ) -> Dict[str, EvaluationResult]:  
        """  
        Evaluate a RAG response.  
          
        Args:  
            query: User query  
            answer: Generated answer  
            context: Retrieved context documents  
            reference_answer: Optional ground truth answer  
              
        Returns:  
            Dictionary of evaluation results by metric  
        """  
        results = {}  
          
        # Relevance  
        results["relevance"] = self._evaluate_relevance(query, answer)  
          
        # Faithfulness  
        results["faithfulness"] = self._evaluate_faithfulness(answer, context)  
          
        # Completeness  
        results["completeness"] = self._evaluate_completeness(query, answer)  
          
        # Coherence  
        results["coherence"] = self._evaluate_coherence(answer)  
          
        # If reference answer provided, compute similarity  
        if reference_answer:  
            results["accuracy"] = self._evaluate_accuracy(answer, reference_answer)  
          
        return results  
      
    def _evaluate_relevance(self, query: str, answer: str) -> EvaluationResult:  
        """Evaluate answer relevance to query."""  
        self._ensure_initialized()  
          
        if not self._llm:  
            return self._heuristic_relevance(query, answer)  
          
        try:  
            from langchain_core.messages import SystemMessage, HumanMessage  
              
            prompt = f"""Rate how relevant this answer is to the question on a scale of 0-1.  
  
Question: {query}  
  
Answer: {answer}  
  
Respond with ONLY a number between 0 and 1."""  
              
            response = self._llm.invoke([  
                SystemMessage(content="You are an evaluation assistant."),  
                HumanMessage(content=prompt)  
            ])  
              
            score = self._extract_score(response.content)  
              
            return EvaluationResult(  
                metric="relevance",  
                score=score,  
                details={"method": "llm"}  
            )  
              
        except Exception as e:  
            logger.error(f"LLM relevance evaluation failed: {e}")  
            return self._heuristic_relevance(query, answer)  
      
    def _heuristic_relevance(self, query: str, answer: str) -> EvaluationResult:  
        """Heuristic relevance evaluation."""  
        query_words = set(query.lower().split())  
        answer_words = set(answer.lower().split())  
          
        # Remove common words  
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why"}  
        query_words -= stopwords  
        answer_words -= stopwords  
          
        if not query_words:  
            return EvaluationResult("relevance", 0.5, {"method": "heuristic"})  
          
        overlap = len(query_words & answer_words)  
        score = min(1.0, overlap / len(query_words))  
          
        return EvaluationResult(  
            metric="relevance",  
            score=score,  
            details={"method": "heuristic", "word_overlap": overlap}  
        )  
      
    def _evaluate_faithfulness(  
        self,  
        answer: str,  
        context: List[Dict[str, Any]]  
    ) -> EvaluationResult:  
        """Evaluate if answer is grounded in context."""  
        if not context:  
            return EvaluationResult(  
                metric="faithfulness",  
                score=0.0,  
                details={"reason": "No context provided"}  
            )  
          
        # Combine context  
        context_text = " ".join([  
            doc.get("content", "") for doc in context  
        ]).lower()  
          
        # Check answer sentences against context  
        sentences = re.split(r'[.!?]+', answer)  
        grounded_count = 0  
          
        for sentence in sentences:  
            sentence = sentence.strip().lower()  
            if not sentence:  
                continue  
              
            # Check if key words from sentence appear in context  
            words = [w for w in sentence.split() if len(w) > 3]  
            if not words:  
                continue  
              
            matches = sum(1 for w in words if w in context_text)  
            if matches / len(words) > 0.3:  
                grounded_count += 1  
          
        total_sentences = len([s for s in sentences if s.strip()])  
        score = grounded_count / max(1, total_sentences)  
          
        return EvaluationResult(  
            metric="faithfulness",  
            score=score,  
            details={  
                "grounded_sentences": grounded_count,  
                "total_sentences": total_sentences  
            }  
        )  
      
    def _evaluate_completeness(self, query: str, answer: str) -> EvaluationResult:  
        """Evaluate if answer addresses the full query."""  
        # Simple heuristic: check if answer is substantial  
        answer_words = len(answer.split())  
          
        if answer_words < 10:  
            score = 0.3  
        elif answer_words < 30:  
            score = 0.6  
        elif answer_words < 100:  
            score = 0.8  
        else:  
            score = 1.0  
          
        # Penalize if answer says "I don't know" type responses  
        uncertain_phrases = [  
            "i don't know", "i'm not sure", "cannot find",  
            "no information", "not available"  
        ]  
          
        answer_lower = answer.lower()  
        for phrase in uncertain_phrases:  
            if phrase in answer_lower:  
                score *= 0.5  
                break  
          
        return EvaluationResult(  
            metric="completeness",  
            score=score,  
            details={"word_count": answer_words}  
        )  
      
    def _evaluate_coherence(self, answer: str) -> EvaluationResult:  
        """Evaluate answer coherence and structure."""  
        # Check for basic coherence indicators  
        score = 1.0  
        issues = []  
          
        # Check sentence structure  
        sentences = re.split(r'[.!?]+', answer)  
        valid_sentences = [s.strip() for s in sentences if s.strip()]  
          
        if len(valid_sentences) == 0:  
            return EvaluationResult(  
                metric="coherence",  
                score=0.0,  
                details={"reason": "No valid sentences"}  
            )  
          
        # Check for incomplete sentences  
        for s in valid_sentences:  
            words = s.split()  
            if len(words) < 3:  
                score -= 0.1  
                issues.append("Short sentence")  
          
        # Check for repetition  
        word_list = answer.lower().split()  
        if len(word_list) > 10:  
            unique_ratio = len(set(word_list)) / len(word_list)  
            if unique_ratio < 0.5:  
                score -= 0.2  
                issues.append("High repetition")  
          
        score = max(0.0, min(1.0, score))  
          
        return EvaluationResult(  
            metric="coherence",  
            score=score,  
            details={"issues": issues, "sentence_count": len(valid_sentences)}  
        )  
      
    def _evaluate_accuracy(  
        self,  
        answer: str,  
        reference: str  
    ) -> EvaluationResult:  
        """Evaluate answer accuracy against reference."""  
        # Simple word overlap metric  
        answer_words = set(answer.lower().split())  
        reference_words = set(reference.lower().split())  
          
        if not reference_words:  
            return EvaluationResult(  
                metric="accuracy",  
                score=0.5,  
                details={"reason": "Empty reference"}  
            )  
          
        intersection = answer_words & reference_words  
        precision = len(intersection) / max(1, len(answer_words))  
        recall = len(intersection) / len(reference_words)  
          
        if precision + recall == 0:  
            f1 = 0.0  
        else:  
            f1 = 2 * precision * recall / (precision + recall)  
          
        return EvaluationResult(  
            metric="accuracy",  
            score=f1,  
            details={"precision": precision, "recall": recall}  
        )  
      
    def _extract_score(self, text: str) -> float:  
        """Extract numeric score from text."""  
        try:  
            # Find first number in text  
            match = re.search(r'(\d+\.?\d*)', text)  
            if match:  
                score = float(match.group(1))  
                return min(1.0, max(0.0, score))  
        except:  
            pass  
        return 0.5  