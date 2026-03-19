"""
Main entry point for HyMem.

This module provides a simple demonstration of the HyMem system.
For evaluation on LoComo dataset, use the scripts/evaluate_locomo.py script.
"""

from hymem.agent import HybridMemAgent
from hymem.config.settings import Settings, LLMConfig, EmbeddingConfig


def simple_demo():
    """
    Simple demonstration of HyMem system.
    
    This function shows basic usage of the HybridMemAgent for
    adding memories and answering questions.
    """
    # Configure your API keys
    # In production, use environment variables or config files
    api_key = "your-api-key-here"
    embed_api_key = "your-embedding-api-key-here"
    
    # Create agent
    agent = HybridMemAgent(
        embed_model="text-embedding-ada-002",
        model_name="gpt-4",
        embed_api_key=embed_api_key,
        api_key=api_key,
        embed_base_url="",
        base_url="",
        backend="openai",
        retrieve_k=15,
        temperature=0.7,
        k_rough=30
    )
    
    # Add some memories
    memories = [
        "Alice works as a software engineer at Google in Mountain View.",
        "Bob is Alice's husband and he works as a doctor at Stanford Hospital.",
        "Alice and Bob have two children: Emma (age 7) and Liam (age 4).",
        "They live in a house in Palo Alto, California.",
        "Last weekend, they visited Yosemite National Park for hiking.",
    ]
    
    print("Adding memories...")
    for memory in memories:
        agent.add_memory(memory)
        print(f"  Added: {memory}")
    
    # Ask questions
    questions = [
        "Where does Alice work?",
        "What does Bob do?",
        "How many children do Alice and Bob have?",
        "Where did they go last weekend?",
    ]
    
    print("\nAnswering questions...")
    for question in questions:
        answer, context = agent.answer_question(question, category=1, answer="")
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Context used: {context[:100]}...")


def demo_with_settings():
    """
    Demonstration using Settings object for configuration.
    """
    # Create settings
    settings = Settings(
        llm=LLMConfig(
            model_name="gpt-4",
            api_key="your-api-key",
            temperature=0.7
        ),
        embedding=EmbeddingConfig(
            model_name="text-embedding-ada-002",
            api_key="your-embedding-api-key"
        ),
        backend="openai"
    )
    
    # Create agent from settings
    agent = HybridMemAgent.from_settings(settings)
    
    # Use agent...
    print(f"Agent created: {agent}")


if __name__ == "__main__":
    print("=" * 50)
    print("HyMem - Hybrid Memory System Demo")
    print("=" * 50)
    
    # Run simple demo (requires API keys)
    # simple_demo()
    
    # Demo with settings
    # demo_with_settings()
    
    print("\nTo run the demo:")
    print("1. Set your API keys in the simple_demo() function")
    print("2. Uncomment the simple_demo() call above")
    print("\nTo evaluate on LoComo dataset:")
    print("  python scripts/evaluate_locomo.py --dataset ./data/locomo10.json")
