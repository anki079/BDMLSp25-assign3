import json

# Load results from file
with open("end_to_end_results.json", "r") as fp:
    results = json.load(fp)

# Display results for each experiment
for experiment, data in results.items():
    print(f"\n{'='*50}")
    print(f"Experiment: {experiment}")
    print(f"{'='*50}")
    
    if "error" in data:
        print(f"Error: {data['error']}")
        continue
    
    # Display timing information
    print("\nTimings:")
    for timing_name, timing_value in data["timings"].items():
        print(f"  {timing_name}: {timing_value:.4f}s")
    
    # Display RAG results
    print("\nRAG Performance:")
    print(f"  Average total time: {data['rag']['avg_total_time']:.4f}s")
    
    print("\nRAG Results per Query:")
    for i, query_result in enumerate(data["rag"]["per_query"]):
        print(f"\n  Query {i+1}: {query_result['question']}")
        print(f"  Answer: {query_result['answer']}")
        print(f"  Retrieval time: {query_result['retrieval_time']:.4f}s")
        print(f"  Generation time: {query_result['generation_time']:.4f}s")
        print(f"  Total time: {query_result['total_time']:.4f}s")
    
    # Display baseline results
    print("\nBaseline Performance (No RAG):")
    print(f"  Average generation time: {data['baseline']['avg_generation_time']:.4f}s")
    
    print("\nBaseline Results per Query:")
    for i, query_result in enumerate(data["baseline"]["per_query"]):
        print(f"\n  Query {i+1}: {query_result['question']}")
        print(f"  Answer: {query_result['answer']}")
        print(f"  Generation time: {query_result['generation_time']:.4f}s")