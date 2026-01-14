from bert_score import score, BERTScorer  
import torch  
from typing import List, Dict, Union, Tuple  
import pandas as pd  
import numpy as np  
  
class RAGBertScoreEvaluator:  
    """  
    BERT Score Evaluator for RAG Pipeline  
    Evaluates similarity between ground truth and predicted paragraphs  
    """  
      
    def __init__(  
        self,   
        model_type: str = "microsoft/deberta-xlarge-mnli",  
        lang: str = "en",  
        rescale_with_baseline: bool = True,  
        device: str = None  
    ):  
        """  
        Initialize the BERT Score Evaluator  
          
        Args:  
            model_type: BERT model to use for scoring  
            lang: Language of the text  
            rescale_with_baseline: Whether to rescale scores with baseline  
            device: Device to run the model on ('cuda' or 'cpu')  
        """  
        self.model_type = model_type  
        self.lang = lang  
        self.rescale_with_baseline = rescale_with_baseline  
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  
          
        # Initialize scorer for batch processing  
        self.scorer = BERTScorer(  
            model_type=model_type,  
            lang=lang,  
            rescale_with_baseline=rescale_with_baseline,  
            device=self.device  
        )  
          
        print(f"BERT Score Evaluator initialized with model: {model_type}")  
        print(f"Running on device: {self.device}")  
      
    def evaluate_single(  
        self,   
        ground_truth: str,   
        prediction: str  
    ) -> Dict[str, float]:  
        """  
        Evaluate a single ground truth and prediction pair  
          
        Args:  
            ground_truth: The reference/ground truth paragraph  
            prediction: The predicted/generated paragraph  
              
        Returns:  
            Dictionary with precision, recall, and F1 scores  
        """  
        P, R, F1 = self.scorer.score([prediction], [ground_truth])  
          
        return {  
            'precision': P.item(),  
            'recall': R.item(),  
            'f1': F1.item()  
        }  
      
    def evaluate_batch(  
        self,   
        ground_truths: List[str],   
        predictions: List[str]  
    ) -> Dict[str, Union[List[float], Dict[str, float]]]:  
        """  
        Evaluate a batch of ground truth and prediction pairs  
          
        Args:  
            ground_truths: List of reference/ground truth paragraphs  
            predictions: List of predicted/generated paragraphs  
              
        Returns:  
            Dictionary with individual scores and aggregate statistics  
        """  
        if len(ground_truths) != len(predictions):  
            raise ValueError("Number of ground truths must match number of predictions")  
          
        P, R, F1 = self.scorer.score(predictions, ground_truths)  
          
        precision_list = P.tolist()  
        recall_list = R.tolist()  
        f1_list = F1.tolist()  
          
        return {  
            'individual_scores': {  
                'precision': precision_list,  
                'recall': recall_list,  
                'f1': f1_list  
            },  
            'aggregate_scores': {  
                'mean_precision': np.mean(precision_list),  
                'mean_recall': np.mean(recall_list),  
                'mean_f1': np.mean(f1_list),  
                'std_precision': np.std(precision_list),  
                'std_recall': np.std(recall_list),  
                'std_f1': np.std(f1_list),  
                'min_f1': np.min(f1_list),  
                'max_f1': np.max(f1_list)  
            }  
        }  
      
    def evaluate_with_details(  
        self,   
        ground_truths: List[str],   
        predictions: List[str],  
        queries: List[str] = None  
    ) -> pd.DataFrame:  
        """  
        Evaluate and return detailed results as a DataFrame  
          
        Args:  
            ground_truths: List of reference paragraphs  
            predictions: List of predicted paragraphs  
            queries: Optional list of queries for context  
              
        Returns:  
            DataFrame with detailed evaluation results  
        """  
        results = self.evaluate_batch(ground_truths, predictions)  
          
        df_data = {  
            'ground_truth': ground_truths,  
            'prediction': predictions,  
            'precision': results['individual_scores']['precision'],  
            'recall': results['individual_scores']['recall'],  
            'f1': results['individual_scores']['f1']  
        }  
          
        if queries:  
            df_data['query'] = queries  
          
        df = pd.DataFrame(df_data)  
          
        # Add quality labels based on F1 score  
        df['quality'] = pd.cut(  
            df['f1'],   
            bins=[-float('inf'), 0.5, 0.7, 0.85, float('inf')],  
            labels=['Poor', 'Fair', 'Good', 'Excellent']  
        )  
          
        return df  
      
    def get_summary_report(  
        self,   
        ground_truths: List[str],   
        predictions: List[str]  
    ) -> str:  
        """  
        Generate a summary report of the evaluation  
          
        Args:  
            ground_truths: List of reference paragraphs  
            predictions: List of predicted paragraphs  
              
        Returns:  
            Formatted string report  
        """  
        results = self.evaluate_batch(ground_truths, predictions)  
        agg = results['aggregate_scores']  
          
        report = f"""  
========================================  
    RAG Pipeline BERT Score Report  
========================================  
  
Total Samples Evaluated: {len(ground_truths)}  
  
Aggregate Scores:  
-----------------  
Metric      | Mean    | Std Dev | Min     | Max  
------------|---------|---------|---------|--------  
Precision   | {agg['mean_precision']:.4f}  | {agg['std_precision']:.4f}  | -       | -  
Recall      | {agg['mean_recall']:.4f}  | {agg['std_recall']:.4f}  | -       | -  
F1 Score    | {agg['mean_f1']:.4f}  | {agg['std_f1']:.4f}  | {agg['min_f1']:.4f}  | {agg['max_f1']:.4f}  
  
Quality Distribution:  
--------------------  
"""  
        # Calculate quality distribution  
        f1_scores = results['individual_scores']['f1']  
        poor = sum(1 for f in f1_scores if f < 0.5)  
        fair = sum(1 for f in f1_scores if 0.5 <= f < 0.7)  
        good = sum(1 for f in f1_scores if 0.7 <= f < 0.85)  
        excellent = sum(1 for f in f1_scores if f >= 0.85)  
          
        total = len(f1_scores)  
        report += f"Poor (F1 < 0.5):      {poor} ({100*poor/total:.1f}%)\n"  
        report += f"Fair (0.5 ≤ F1 < 0.7): {fair} ({100*fair/total:.1f}%)\n"  
        report += f"Good (0.7 ≤ F1 < 0.85): {good} ({100*good/total:.1f}%)\n"  
        report += f"Excellent (F1 ≥ 0.85): {excellent} ({100*excellent/total:.1f}%)\n"  
        report += "========================================"  
          
        return report  
  
  
# Standalone function for quick evaluation  
def quick_bert_score(  
    ground_truths: Union[str, List[str]],   
    predictions: Union[str, List[str]],  
    model_type: str = "microsoft/deberta-xlarge-mnli"  
) -> Dict:  
    """  
    Quick BERT score evaluation without initializing a class  
      
    Args:  
        ground_truths: Single string or list of reference paragraphs  
        predictions: Single string or list of predicted paragraphs  
        model_type: BERT model to use  
          
    Returns:  
        Dictionary with scores  
    """  
    if isinstance(ground_truths, str):  
        ground_truths = [ground_truths]  
        predictions = [predictions]  
      
    P, R, F1 = score(  
        predictions,   
        ground_truths,   
        model_type=model_type,  
        lang="en",  
        rescale_with_baseline=True  
    )  
      
    return {  
        'precision': P.tolist(),  
        'recall': R.tolist(),  
        'f1': F1.tolist(),  
        'mean_f1': F1.mean().item()  
    }  

# Example 1: Quick single evaluation  
if __name__ == "__main__":  
      
    # Initialize evaluator  
    evaluator = RAGBertScoreEvaluator(  
        model_type=
        # "roberta-base",  
        "microsoft/deberta-xlarge-mnli",  # Best for accuracy  
        # model_type="roberta-large",  # Faster alternative  
    )  
      
    # Single evaluation  
    ground_truth = """  
    The primary objective of this thesis is to develop an end-to-end pipeline for scenario extraction specifically designed for off-highway autonomous vehicle testing. This is achieved by systematically integrating various computer vision (CV) tasks and internally fusing them with Large Vision-Language Models (LVLMs) to generate a comprehensive object list containing the metadata necessary for robust scenario generation.
    The research focus is categorized into two interconnected areas:
    1. Computer Vision Enhancement
    The researcher aims to evaluate and integrate various AI models to handle core perception tasks, including:
    • Object Detection: Identifying and tracking road actors.
    • Depth Estimation: Inferring distances from single RGB frames.
    • Speed Estimation: Determining the velocity of the ego vehicle and surrounding objects.
    • Global Positioning: Applying standard mathematical formulations to calculate the global positions of both ego and non-ego objects to provide metric-accurate spatial representations.
    2. Computer Vision and LVLM Fusion
    A central goal is to bridge the gap between CV and LVLMs by creating a fusion methodology that leverages their complementary strengths. While traditional computer vision provides perceptual capability and metric precision, LVLMs provide contextual understanding and semantic reasoning. This fusion is used to address specific tasks that require a deeper understanding of the scene, such as:
    • Conflict Resolution: Resolving instances where different detection layers assign conflicting class labels to the same object.
    • Label Refinement: Improving generic class labels to more specific categorizations.
    • False Positive Verification: Systematically evaluating detected objects to determine their validity and filtering out errors.
    • Motion Classification: Categorizing objects as "static" or "dynamic" by combining kinematic data with semantic reasoning (e.g., knowing a tree is inherently static despite sensor noise).
    • Orientation Estimation: Determining which way an object is facing (towards, away, left, or right) relative to the ego vehicle.
    The Central Research Question
    The thesis is guided by the following question: "How can inference-based computer vision methods, combined with Large Vision Language Models, enable robust, adaptable, and scalable scenario extraction for autonomous vehicles in complex off-highway environments?".  
    """  
      
    prediction = """  
    
    The research objectives of the thesis are outlined in Document 1 (page 36). The central aim is to bridge computer vision and large vision-language models by developing a fusion methodology that allows both domains to work together, thereby improving scenario extraction. Specifically, the objectives are divided into two interconnected areas:

    1. **Computer Vision Enhancement**:
    - Evaluating and integrating various AI models for core perception tasks, including object detection, depth estimation, and speed estimation.

    2. **Fusion of Computer Vision and Vision-Language Models**:
    - Creating a methodology that enables the synergy between computer vision and vision-language models, leading to the generation of comprehensive object lists containing all relevant metadata from the pipeline.

    As stated in the document:
    "The central idea behind this research focuses on bridging computer vision and large vision language models by creating a fusion methodology that enables both domains to work synergistically, thereby improving scenario extraction through the generation of comprehensive object lists containing all relevant metadata from the pipeline." (Document 1, p. 36)

    In summary, the thesis aims to enhance computer vision tasks and integrate them with vision-language models to improve scenario extraction for autonomous systems.  
    """ 

      
    # Evaluate single pair  
    single_result = evaluator.evaluate_single(ground_truth, prediction)  
    print("Single Evaluation Result:")  
    print(f"  Precision: {single_result['precision']:.4f}")  
    print(f"  Recall: {single_result['recall']:.4f}")  
    print(f"  F1 Score: {single_result['f1']:.4f}")  