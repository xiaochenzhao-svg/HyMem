"""
Evaluation script for HyMem on LoComo dataset.

This script evaluates the HybridMemAgent on the LoComo benchmark,
which tests long-context memory capabilities across multiple sessions.
"""

import os
import argparse
import logging
from typing import Optional, List
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pickle
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hymem.agent import HybridMemAgent
from hymem.data.loader import load_locomo_dataset, LoCoMoSample


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def process_conversation(
    agent: HybridMemAgent,
    sample: LoCoMoSample,
    logger: logging.Logger,
    max_workers: int = 8
) -> None:

    def build_session_text(session):
        session_temp = f"Session starts at: {session.date_time}\n"

        for turn in session.turns:
            conversation_tmp = f"Speaker {turn.speaker} says : {turn.text}"
            if hasattr(turn, "blip_caption") and turn.blip_caption is not None:
                conversation_tmp += (
                    " (There is a picture next to them. Image name: "
                    + str(turn.query)
                    + ". Description of the image content: "
                    + str(turn.blip_caption)
                    + ")"
                )
            session_temp += conversation_tmp + "\n"

        return session_temp

    def worker(session):
        content = build_session_text(session)

        summary = []
        max_retries = 2

        for _ in range(max_retries + 1):
            summary = agent.memory_system.analyze_content(
                content,
                agent.memory_system.llm_controller
            )
            if summary:
                break

        if not summary:
            return None

        return (content, summary, session.date_time)
    sessions = list(sample.conversation.sessions.values())
    results = []


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, s) for s in sessions]
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res:
                results.append(res)

    for content, summary, time in results:
        agent.add_memory(content, time=time, precomputed_summary=summary)


def load_cached_memories(
    agent: HybridMemAgent,
    cache_dir: str,
    sample_idx: int,
    embed_model: str,
    embed_api_key: str,
    embed_base_url: str,
    logger: logging.Logger
) -> bool:
    """
    Load cached memories for a sample.
    
    Args:
        agent: HybridMemAgent instance
        cache_dir: Directory containing cached files
        sample_idx: Sample index
        embed_model: Embedding model name
        embed_api_key: API key for embedding model
        embed_base_url: Base URL for embedding model
        logger: Logger instance
        
    Returns:
        True if cache loaded successfully, False otherwise
    """
    memory_cache_file = os.path.join(
        cache_dir, f"memory_cache_sample_{sample_idx}.pkl"
    )
    
    if not os.path.exists(memory_cache_file):
        return False
    
    logger.info(f"Loading cached memories for sample {sample_idx}")
    
    # Load memories
    with open(memory_cache_file, 'rb') as f:
        cached_memories = pickle.load(f)
    agent.memory_system.memories = cached_memories
    
    # Load summaries
    summary_cache_file = os.path.join(
        cache_dir, f"summary_list_cache_sample_{sample_idx}.pkl"
    )
    if os.path.exists(summary_cache_file):
        with open(summary_cache_file, 'rb') as f:
            cached_summary_list = pickle.load(f)
        agent.memory_system.summary_list = cached_summary_list
    else:
        agent.memory_system.summary_list = []
        logger.warning("No summary list found in cache")
    
    # Load retriever
    retriever_cache_file = os.path.join(
        cache_dir, f"retriever_cache_sample_{sample_idx}.pkl"
    )
    retriever_embeddings_file = os.path.join(
        cache_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
    )
    
    if os.path.exists(retriever_cache_file):
        agent.memory_system.retriever.load(retriever_cache_file, retriever_embeddings_file)
    else:
        logger.info("Rebuilding retriever from cached memories")
        agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
            cached_memories, embed_model, embed_api_key, embed_base_url
        )
    
    logger.info(f"Successfully loaded {len(cached_memories)} memories")
    return True


def save_cached_memories(
    agent: HybridMemAgent,
    cache_dir: str,
    sample_idx: int,
    logger: logging.Logger
) -> None:
    """
    Save memories to cache.
    
    Args:
        agent: HybridMemAgent instance
        cache_dir: Directory to save cache files
        sample_idx: Sample index
        logger: Logger instance
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save memories
    memory_cache_file = os.path.join(
        cache_dir, f"memory_cache_sample_{sample_idx}.pkl"
    )
    with open(memory_cache_file, 'wb') as f:
        pickle.dump(agent.memory_system.memories, f)
    
    # Save summaries
    summary_cache_file = os.path.join(
        cache_dir, f"summary_list_cache_sample_{sample_idx}.pkl"
    )
    with open(summary_cache_file, 'wb') as f:
        pickle.dump(agent.memory_system.summary_list, f)
    
    # Save retriever
    retriever_cache_file = os.path.join(
        cache_dir, f"retriever_cache_sample_{sample_idx}.pkl"
    )
    retriever_embeddings_file = os.path.join(
        cache_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
    )
    agent.memory_system.retriever.save(retriever_cache_file, retriever_embeddings_file)
    
    logger.info(f"Cached {len(agent.memory_system.memories)} memories")


def evaluate_dataset(
    dataset_path: str,
    output_path: Optional[str],
    ratio: float,
    backend: str,
    temperature: float,
    retrieve_k: int,
    embed_model: str,
    model_name: str,
    embed_api_key: str,
    api_key: str,
    embed_base_url: str,
    base_url: str,
    log_name: str,
    retrieve_k_rough: int
) -> None:
    """
    Evaluate the dataset using specified parameters.
    
    Args:
        dataset_path: Path to dataset file
        output_path: Optional path for output results
        ratio: Ratio of dataset to evaluate
        backend: Backend type
        temperature: Temperature for generation
        retrieve_k: Number of results for retrieval
        embed_model: Embedding model name
        model_name: LLM model name
        embed_api_key: API key for embedding
        api_key: API key for LLM
        embed_base_url: Base URL for embedding
        base_url: Base URL for LLM
        log_name: Log file name
        retrieve_k_rough: Number of results for rough retrieval
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"{log_name}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)
    
    logger.info(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Apply ratio if specified
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # Setup cache directory
    cache_dir = os.path.join(
        os.path.dirname(__file__),
        "cached_memories_advanced_{}_{}".format(model_name, temperature)
    )
    
    # Statistics
    total_questions = 0
    category_counts = defaultdict(int)
    allowed_categories = [1, 2, 3, 4]

    # Process each sample
    for sample_idx, sample in enumerate(tqdm(samples, total=len(samples))):
        # Create agent for this sample
        agent = HybridMemAgent(
            embed_model=embed_model,
            model_name=model_name,
            embed_api_key=embed_api_key,
            api_key=api_key,
            embed_base_url=embed_base_url,
            base_url=base_url,
            backend=backend,
            retrieve_k=retrieve_k,
            temperature=temperature,
            k_rough=retrieve_k_rough
        )
        
        # Try to load cached memories
        cache_loaded = load_cached_memories(
            agent, cache_dir, sample_idx,
            embed_model, embed_api_key, embed_base_url, logger
        )
        
        if not cache_loaded:
            logger.info(f"Creating new memories for sample {sample_idx}")
            process_conversation(agent, sample, logger)
            save_cached_memories(agent, cache_dir, sample_idx, logger)
        
        logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")
        valid_qas = [
            (i, qa)
            for i, qa in enumerate(
                [qa for qa in sample.qa if int(qa.category) in allowed_categories]
            )
        ]

        results = [None] * len(valid_qas)

        def worker(idx, qa):
            prediction, user_prompt = agent.answer_question(
                qa.question, qa.category, qa.final_answer
            )
            return (
                idx,
                qa,
                prediction,
                user_prompt
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i, qa) for i, qa in valid_qas]

            for f in tqdm(as_completed(futures), total=len(futures), desc="questions", leave=False):
                idx, qa, prediction, user_prompt = f.result()
                results[idx] = (
                    qa,
                    prediction,
                    user_prompt
                )

        for res in results:
            if res is None:
                continue
            qa, prediction, user_prompt = res
            total_questions += 1
            category_counts[qa.category] += 1
            logger.info(f"\nQuestion {total_questions}: {qa.question}")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Reference: {qa.final_answer}")
            logger.info(f"User Prompt: {user_prompt}")
            logger.info(f"Category: {qa.category}")
        del agent
        gc.collect()
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Categories: {dict(category_counts)}")



def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate HyMem on LoComo dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/locomo10.json",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Ratio of dataset to evaluate (0.0 to 1.0)"
    )
    
    # Model arguments
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="Backend to use (openai)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for the model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4.1-mini",
        help="Name of the LLM model"
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="text-embedding-3-small",
        help="Name of the embedding model"
    )
    
    # API arguments
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for LLM"
    )
    parser.add_argument(
        "--embed_api_key",
        type=str,
        default="",
        help="API key for embedding model"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Base URL for LLM API"
    )
    parser.add_argument(
        "--embed_base_url",
        type=str,
        default="",
        help="Base URL for embedding model API"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--retrieve_k",
        type=int,
        default=15,
        help="Number of results for retrieval"
    )
    parser.add_argument(
        "--retrieve_k_rough",
        type=int,
        default=30,
        help="Number of results for rough retrieval"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_name",
        type=str,
        default="locomo",
        help="Log file name prefix"
    )
    
    args = parser.parse_args()
    
    # Validate ratio
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # Resolve paths
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, args.dataset)
    
    output_path = None
    if args.output:
        output_path = os.path.join(script_dir, args.output)
    
    # Run evaluation
    evaluate_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        ratio=args.ratio,
        backend=args.backend,
        temperature=args.temperature,
        retrieve_k=args.retrieve_k,
        embed_model=args.embed_model,
        model_name=args.model_name,
        embed_api_key=args.embed_api_key,
        api_key=args.api_key,
        embed_base_url=args.embed_base_url,
        base_url=args.base_url,
        log_name=args.log_name,
        retrieve_k_rough=args.retrieve_k_rough
    )


if __name__ == "__main__":
    main()
