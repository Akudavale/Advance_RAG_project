"""  
src/evaluation/feedback_loop.py  
-------------------------------  
Feedback collection and learning loop for RAG improvement.  
"""  
  
import logging  
from typing import Dict, List, Any, Optional  
from datetime import datetime  
from dataclasses import dataclass, field  
  
logger = logging.getLogger(__name__)  
  
  
@dataclass  
class Feedback:  
    """User feedback on a response."""  
    feedback_id: str  
    query: str  
    answer: str  
    rating: int  # 1-5  
    comment: Optional[str] = None  
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  
    metadata: Dict[str, Any] = field(default_factory=dict)  
  
  
class FeedbackLoop:  
    """  
    Collects and processes user feedback for RAG improvement.  
      
    Features:  
    - Feedback collection  
    - Pattern analysis  
    - Improvement suggestions  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the feedback loop.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self._feedback_store: List[Feedback] = []  
        self._feedback_id_counter = 0  
      
    def submit_feedback(  
        self,  
        query: str,  
        answer: str,  
        rating: int,  
        comment: Optional[str] = None,  
        metadata: Optional[Dict[str, Any]] = None  
    ) -> Dict[str, Any]:  
        """  
        Submit user feedback.  
          
        Args:  
            query: Original query  
            answer: Generated answer  
            rating: Rating 1-5  
            comment: Optional comment  
            metadata: Optional metadata  
              
        Returns:  
            Feedback submission result  
        """  
        # Validate rating  
        rating = max(1, min(5, rating))  
          
        # Create feedback  
        self._feedback_id_counter += 1  
        feedback = Feedback(  
            feedback_id=f"fb_{self._feedback_id_counter}",  
            query=query,  
            answer=answer,  
            rating=rating,  
            comment=comment,  
            metadata=metadata or {}  
        )  
          
        self._feedback_store.append(feedback)  
          
        logger.info(f"Feedback submitted: {feedback.feedback_id}, rating={rating}")  
          
        return {  
            "status": "success",  
            "feedback_id": feedback.feedback_id,  
            "message": "Feedback recorded"  
        }  
      
    def get_feedback_summary(self) -> Dict[str, Any]:  
        """Get summary of collected feedback."""  
        if not self._feedback_store:  
            return {"message": "No feedback collected"}  
          
        ratings = [f.rating for f in self._feedback_store]  
          
        return {  
            "total_feedback": len(self._feedback_store),  
            "average_rating": sum(ratings) / len(ratings),  
            "rating_distribution": {  
                i: ratings.count(i) for i in range(1, 6)  
            },  
            "positive_count": sum(1 for r in ratings if r >= 4),  
            "negative_count": sum(1 for r in ratings if r <= 2)  
        }  
      
    def get_low_rated_queries(self, threshold: int = 2) -> List[Dict[str, Any]]:  
        """Get queries with low ratings for analysis."""  
        low_rated = [  
            {  
                "query": f.query,  
                "answer": f.answer[:200] + "..." if len(f.answer) > 200 else f.answer,  
                "rating": f.rating,  
                "comment": f.comment,  
                "timestamp": f.timestamp  
            }  
            for f in self._feedback_store  
            if f.rating <= threshold  
        ]  
          
        return low_rated  
      
    def analyze_patterns(self) -> Dict[str, Any]:  
        """Analyze feedback patterns for improvement insights."""  
        if len(self._feedback_store) < 5:  
            return {"message": "Need more feedback for pattern analysis"}  
          
        # Analyze by query length  
        short_query_ratings = []  
        long_query_ratings = []  
          
        for f in self._feedback_store:  
            if len(f.query.split()) < 10:  
                short_query_ratings.append(f.rating)  
            else:  
                long_query_ratings.append(f.rating)  
          
        # Analyze by answer length  
        short_answer_ratings = []  
        long_answer_ratings = []  
          
        for f in self._feedback_store:  
            if len(f.answer.split()) < 50:  
                short_answer_ratings.append(f.rating)  
            else:  
                long_answer_ratings.append(f.rating)  
          
        def avg(lst):  
            return sum(lst) / len(lst) if lst else 0  
          
        return {  
            "query_length_analysis": {  
                "short_queries_avg_rating": avg(short_query_ratings),  
                "long_queries_avg_rating": avg(long_query_ratings)  
            },  
            "answer_length_analysis": {  
                "short_answers_avg_rating": avg(short_answer_ratings),  
                "long_answers_avg_rating": avg(long_answer_ratings)  
            },  
            "total_analyzed": len(self._feedback_store)  
        }  