def formulate_prompt(query, docs):
    assert len(docs) != 0, "There's no any context documents."
    num_docs = len(docs)
    
    prompt = f"""
        You are an expert question answering system, 
        I'll give you question and context and you'll return the answer.
        If the answer is not in the contexts below, please response 'I don't know the answer'.
        Query : {query} 
    """

    for id in range(num_docs):
        prompt += f"\n Relevant context passage {id}: {docs[id]}"
        
    return prompt