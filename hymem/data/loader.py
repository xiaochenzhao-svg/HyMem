"""
Dataset loading utilities for HyMem.

This module provides data structures and functions for loading and parsing
the LoComo dataset, which contains multi-session conversations and QA pairs.
"""

import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QA:
    """
    Question-Answer pair with metadata.
    
    Attributes:
        question: The question text
        answer: The standard answer
        evidence: List of evidence sources
        category: Question category (1-5)
        adversarial_answer: Adversarial answer for category 5 questions
    """
    
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """
        Get the appropriate answer based on category.
        
        Returns:
            Adversarial answer for category 5, standard answer otherwise
        """
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


@dataclass
class Turn:
    """
    A single turn in a conversation.
    
    Attributes:
        speaker: Speaker identifier (e.g., "A", "B")
        text: The text content of the turn
    """
    
    speaker: str
    text: str


@dataclass
class Session:
    """
    A conversation session containing multiple turns.
    
    Attributes:
        session_id: Unique session identifier
        date_time: Timestamp of the session
        turns: List of conversation turns in this session
    """
    
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    """
    A complete conversation with multiple sessions.
    
    Attributes:
        speaker_a: Identifier for speaker A
        speaker_b: Identifier for speaker B
        sessions: Dictionary mapping session IDs to Session objects
    """
    
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]


@dataclass
class LoCoMoSample:
    """
    A single sample from the LoComo dataset.
    
    Attributes:
        sample_id: Unique identifier for this sample
        qa: List of question-answer pairs
        conversation: The conversation associated with this sample
    """
    
    sample_id: str
    qa: List[QA]
    conversation: Conversation


def parse_session(
    session_data: List[dict],
    session_id: int,
    date_time: str
) -> Session:
    """
    Parse a single session's data from JSON format.
    
    Handles turns with images by using their captions.
    
    Args:
        session_data: List of turn dictionaries from JSON
        session_id: Session identifier
        date_time: Timestamp string
        
    Returns:
        Parsed Session object
    """
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
        
        turns.append(Turn(
            speaker=turn["speaker"],
            text=text
        ))
    
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def parse_conversation(conv_data: dict) -> Conversation:
    """
    Parse conversation data from JSON format.
    
    Args:
        conv_data: Conversation data dictionary from JSON
        
    Returns:
        Parsed Conversation object
    """
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.turns:
                    sessions[session_id] = session
    
    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )


def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file.
    
    Handles image-based content by using captions.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        List of LoCoMoSample objects containing the parsed data
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []
    
    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0
            
            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1
                    
                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
            
            # Parse conversation
            conversation = parse_conversation(sample["conversation"])
            
            # Create sample object
            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation
            )
            samples.append(sample_obj)
            
            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)
            
            # Print statistics for this sample
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")
    
    return samples


def get_dataset_statistics(samples: List[LoCoMoSample]) -> Dict:
    """
    Get basic statistics about the text-only dataset.
    
    Args:
        samples: List of LoCoMoSample objects
        
    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_samples": len(samples),
        "total_qa_pairs": sum(len(sample.qa) for sample in samples),
        "total_sessions": sum(len(sample.conversation.sessions) for sample in samples),
        "total_turns": sum(
            sum(len(session.turns) for session in sample.conversation.sessions.values())
            for sample in samples
        ),
        "qa_with_adversarial": sum(
            sum(1 for qa in sample.qa if qa.adversarial_answer is not None)
            for sample in samples
        )
    }
    return stats
