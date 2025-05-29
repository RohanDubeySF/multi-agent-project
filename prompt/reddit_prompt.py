from langchain.prompts import PromptTemplate

youtube_reddit_prompt=PromptTemplate(
    template="""You are helping a user create a Reddit post to share something they recently learned.

    Given a youtube video summary, write a Reddit post that:

    - Starts by explaining that the user recently learned or explored the topic.
    - Summarizes the key concepts or takeaways in clear, digestible language.
    - Explains why the user is sharing this — e.g., to document their learning or contribute to the community.
    - Ends with a question, prompt, or invitation for feedback or resource suggestions.
    - Uses a neutral, informative, and conversational tone that fits Reddit communities like r/learnmachinelearning or r/coding.

    Here is the summary of the YouTube video:
    {summary}

    Write the Reddit post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "Reddit<Reddit logo>:\n<Your Reddit post here>\n\n\n\n""",
    input_variables=["summary"]
)

article_reddit_prompt=PromptTemplate(
    template="""You are helping a user create a Reddit post to share something they recently learned.

    Given a article summary, write a Reddit post that:

    - Starts by explaining that the user recently learned or explored the topic.
    - Summarizes the key concepts or takeaways in clear, digestible language.
    - Explains why the user is sharing this — e.g., to document their learning or contribute to the community.
    - Ends with a question, prompt, or invitation for feedback or resource suggestions.
    - Uses a neutral, informative, and conversational tone that fits Reddit communities like r/learnmachinelearning or r/coding.

    Article Summary:
    {summary}

    Write the Reddit post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "Reddit<Reddit logo>:\n\n <Your Reddit post here>\n\n\n\n""",
    input_variables=["summary"])