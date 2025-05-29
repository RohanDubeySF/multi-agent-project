from langchain.prompts import PromptTemplate

youtube_twitter_prompt=PromptTemplate(
    template="""You are a professional social media content creator.

    Given the summary of a YouTube video, write a Twitter post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the video in simple terms.
    - Keeps the post under 200 characters.
    - Ends with a relevant call-to-action, that encourages reader engagement or discussion.
    - Also include the link of the video

    Here is the summary of the YouTube video:
    {summary}

   Write the tweet in first person, as if the user is sharing their own learning.
    output Format:\n
    "Twitter<twitter logo>:\n\n <Your Twitter post here>\n\n\n\n""",
    input_variables=["summary"])

article_twitter_prompt=PromptTemplate(
    template="""You are a professional social media content creator.

    Given the summary of an Article, write a Twitter post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the Article in simple terms.
    - Keeps the post under 200 characters.
    - Ends with a relevant call-to-action, that encourages reader engagement or discussion.

    Here is the summary of the Article:
    {summary}

    Write the tweet in first person, as if the user is sharing their own learning.
    output Format:\n
    "Twitter<twitter logo>:\n\n <Your Twitter post here>\n\n\n\n""",
    input_variables=["summary"])