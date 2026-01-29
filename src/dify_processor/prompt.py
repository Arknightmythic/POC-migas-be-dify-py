class Prompt:
    def __init__(self):
        print("Prompt Initialized")

    def chunk_topic(self, text_chunk):
        return f"""Analyze the following text chunk and provide a concise subtitle/topic that best describes the main subject of this text. 
Respond with only the topic/subtitle, no additional text or formatting.
Use the same language as the text provided, defaulting in Bahasa Indonesia.

Text chunk:
{text_chunk[:1000]}...

Topic:"""
    
    def chunk_description(self, text_chunk):
        return f"""Analyze the following text chunk and provide a brief description (1-2 sentences) of what this text is talking about.
Focus on the main points and key information contained in this chunk.
Use the same language as the text provided, defaulting in Bahasa Indonesia.

Text chunk:
{text_chunk[:1000]}...

Description:"""