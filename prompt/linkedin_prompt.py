from langchain.prompts import PromptTemplate

youtube_linkedin_prompt=PromptTemplate(
    template="""You are a professional social media content creator.

    Given the summary of a YouTube video, write a LinkedIn post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the video.
    - Maintains a professional and informative tone suitable for a tech-savvy audience
    - Includes 2-3 short bullet points or takeaways.
    - Ends with a relevant call-to-action, like "What are your thoughts?" or "Have you tried this yet?"

    Here is the summary of the YouTube video:
    {summary}

    Write the LinkedIn post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "LinkedIn<LinkedIn logo>:\n<Your LinkedIn post here>\n\n\n\n""",
    input_variables=["summary"]
)






article_linkedin_prompt=PromptTemplate(
    template="""You are a professional social media content creator.

    Given the summary of an Article, write a LinkedIn post that:
    - Starts by stating that the person recently learned or explored this topic.
    - Highlights the key value or insight from the Article
    - Maintains a professional and informative tone suitable for a tech-savvy audience
    - Includes 2-3 short bullet points or takeaways.
    - Ends with a relevant call-to-action, that encourages reader engagement or discussion.
    - strictly follow the above content guidelines and dont mention any other thing which is not present in the context 
    - Dont mention for any youtube video link

    Here is the summary of the Article:
    {summary}

    Write the LinkedIn post in first person, as if the user is sharing their own learning experience.
    output Format:\n
    "LinkedIn<LinkedIn logo>:\n<Your LinkedIn post here>\n\n\n\n""",
    input_variables=["summary"])