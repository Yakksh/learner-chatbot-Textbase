import os
from textbase import bot, Message
from textbase.models import OpenAI
from typing import List
import nltk
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Load your OpenAI API key
OpenAI.api_key = ""
# or from environment variable:
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def sumy_text_rank_summarize(text_content, percent):
    # TextRank is an unsupervised text summarization technique that uses the intuition behind the PageRank algorithm.
    # Initializing the parser
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    # Initialize the stemmer
    stemmer = Stemmer('english')
    # Initializing the summarizer
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    # Finding number of sentences and applying percentage on it: since sumy requires number of lines
    sentence_token = sent_tokenize(text_content)
    select_length = int(len(sentence_token) * (int(percent) / 100))

    # Evaluating and saving the Summary
    summary = ""
    for sentence in summarizer(parser.document, sentences_count=select_length):
        summary += str(sentence)
    # Returning NLTK Summarization Output
    return summary

url = "https://www.youtube.com/watch?v=iuYlGRnC7J8"
video_id = url.split('=')[-1]
percent = 10
formatter = TextFormatter()
transcript = YouTubeTranscriptApi.get_transcript(video_id)
formatted_text = formatter.format_transcript(transcript).replace("\n", " ")
transcribe = sumy_text_rank_summarize(formatted_text, percent)

# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT = f"""Act as an Instructor and understand the given text as the user will be questioning you based on this text.
Make sure you are ready to answer related to the given text.
Text: {transcribe}
This text is actaully an transcript of an youtube video so, treat it like an youtube video.
When the user say Hi, Hello, or something similar to this you will be responding with an two lines about the text.
Also come up with the title of the Text as this two lines should contain the title of the text that you came
up with and urging the user that they can ask anything related to the Text.
"""

@bot()
def on_message(message_history: List[Message], state: dict = None):
    # Generate GPT-3.5 Turbo response
    bot_response = OpenAI.generate(
        system_prompt=SYSTEM_PROMPT,
        message_history=message_history, # Assuming history is the list of user messages
        model="gpt-3.5-turbo",
    )

    response = {
        "data": {
            "messages": [
                {
                    "data_type": "STRING",
                    "value": bot_response
                }
            ],
            "state": state
        },
        "errors": [
            {
                "message": ""
            }
        ]
    }

    return {
        "status_code": 200,
        "response": response
    }